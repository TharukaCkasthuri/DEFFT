

class FedHiKoDServer(Server):

    """
    The federated learning server class for FedHiKoD.
    """

    def __init__(self, *args, fitness_cfg: FitnessCfg = FitnessCfg(), **kwargs):
        super().__init__(*args, **kwargs)
        self.fitness_cfg = fitness_cfg

        self._g_ema = None             # torch.Tensor on CPU or None
        self._g_beta = 0.5             # momentum factor for g_EMA
        self._g_num_final_layers = 2   # must match clients' alignment layers
        self._groups: dict[str, int] = {}              # cid -> group id (fixed after round 1)
        self._client_distributions: dict[str, dict[int, float]] = {}
        self._leader_models_by_group: dict[int, dict] = {}  # gid -> state_dict
  

    def flatten_last_modules(self, model: torch.nn.Module, last_n_modules: int | None = None) -> torch.Tensor:
        """
        Flatten parameters from the last `last_n_modules` child modules of `model`.
        If `last_n_modules` is None, flatten all parameters.
        """
        with torch.no_grad():
            # children keep registration order: fc1, relu, fc2
            modules = [m for _, m in model.named_children()]
            # drop non-param modules like ReLU
            modules = [m for m in modules if any(p.requires_grad for p in m.parameters(recurse=False))]

            if last_n_modules is None:
                selected = modules
            else:
                selected = modules[-last_n_modules:]

            flats = []
            for m in selected:
                for p in m.parameters(recurse=False):
                    flats.append(p.detach().cpu().flatten())
            if not flats:  # fallback to all params if something odd happens
                flats = [p.detach().cpu().flatten() for p in model.parameters()]
            return torch.cat(flats)


    def _global_update_vector(self, old_model: torch.nn.Module, new_model: torch.nn.Module,
                              num_final_layers: int | None = 2) -> torch.Tensor:
        with torch.no_grad():
            v_old = self.flatten_last_modules(old_model)
            v_new = self.flatten_last_modules(new_model)
            dtheta = (v_new - v_old).detach().cpu()
        return dtheta
    

    def _update_global_momentum(self, dtheta: torch.Tensor) -> torch.Tensor:
        # no-op if the update is degenerate
        if dtheta is None or torch.norm(dtheta) == 0:
            return self._g_ema

        if self._g_ema is None or torch.numel(self._g_ema) != torch.numel(dtheta):
            # initialize on first use or if shape changed
            self._g_ema = dtheta.clone().detach().cpu()
        else:
            self._g_ema = (self._g_beta * self._g_ema) + ((1.0 - self._g_beta) * dtheta.detach().cpu())
        return self._g_ema


    def _embed_for_clustering(self, distributions: Dict, total_classes: int, alpha: int = 1)-> Tuple[list, np.ndarray]:
        """
        Create a fixed-size embedding vector from each client's class distribution
        and compute pairwise Jensen-Shannon distances.
        """
        from scipy.spatial.distance import pdist
        from scipy.spatial.distance import jensenshannon

        client_ids = list(distributions.keys())
        vectors = []

        for cid in client_ids:
            counts = np.array([distributions[cid].get(i, 0) for i in range(total_classes)], dtype=float)
            counts += alpha  # Laplace smoothing
            counts /= counts.sum()
            vectors.append(counts)

        vectors = np.vstack(vectors)
        jsd_condensed = pdist(vectors, metric=lambda u, v: jensenshannon(u, v))
        return client_ids, jsd_condensed


    def hierarchical_clustering(self, client_ids, jsd_condensed, num_clusters):
        """
        Perform hierarchical clustering using Jensen–Shannon distances.
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from collections import defaultdict

        # Perform linkage using precomputed distances
        Z = linkage(jsd_condensed, method='average', optimal_ordering=True)
        clusters = fcluster(Z, t=num_clusters, criterion='maxclust')

        client_to_cluster = {}
        cluster_to_clients = defaultdict(list)

        for cid, cluster_id in zip(client_ids, clusters):
            client_to_cluster[cid] = cluster_id
            cluster_to_clients[cluster_id].append(cid)

        return client_to_cluster, cluster_to_clients


    def _collect_self_reports(self, client: Client) -> dict[str, dict]:
        cfg = self.fitness_cfg
        recipe_hash = hashlib.sha256(f"metric:loss|clip:{cfg.clip_range}".encode()).hexdigest()

        if not hasattr(client, "report_fitness"):
            raise RuntimeError(f"Client {client.client_id} lacks report_fitness()")
        
        # these are hyperparams
        rpt = client.report_fitness(
            recipe_hash=recipe_hash,
            clip_range=cfg.clip_range,
        )

        return rpt
    

    def _client_group_similarity(self, cid: str) -> float:
        gid = self._c2g[cid]
        client_dist = self._client_distributions[cid]         # dict[int, count]
        group_dist = self._group_distributions[gid]           # dict[int, prob]

        all_classes = sorted(set(client_dist.keys()) | set(group_dist.keys()))
        p = np.array([client_dist.get(c, 0) for c in all_classes], dtype=float)
        q = np.array([group_dist.get(c, 0.0) for c in all_classes], dtype=float)

        p = p / (p.sum() + 1e-8)
        q = q / (q.sum() + 1e-8)

        jsd = jensenshannon(p, q)            # in [0, ~1]
        sim = 1.0 - float(jsd)
        return float(np.clip(sim, 0.0, 1.0))
    

    def _compute_tau_per_client(self, cluster_q: dict[int, float]) -> dict[str, float]:
        tau = {}
        tau_min, tau_max = 0.05, 0.7

        for cid in self.client_dict.keys():
            gid = self._c2g[cid]
            qg = cluster_q.get(gid, 1.0)         # already normalized [0.1, 1]
            sim = self._client_group_similarity(cid)
            raw = qg * sim
            tau[cid] = float(np.clip(raw, tau_min, tau_max))
        return tau


    def _aggregate(self, trained_clients, weights):
        """
        Aggregate the models of the clients using FedHiKoD aggregation strategy.
        """
        return weighted_avg(self.global_model, trained_clients, weights)


    def _cluster_quality(self, reports: dict[str, dict], c2g: dict[str, int]) -> dict[int, float]:
        """
        Aggregate client 'score' values into per-cluster quality.
        Normalizes to [0.1, 1.0] to avoid zero weights.
        """

        cluster_scores = defaultdict(list)
        for cid, rpt in reports.items():
            gid = c2g.get(cid)
            score = float(rpt.get("score", 0.0))
            cluster_scores[gid].append(score)

        if not cluster_scores:
            return {}

        avg_scores = {g: np.mean(v) for g, v in cluster_scores.items()}
        vals = np.array(list(avg_scores.values()))
        vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        vals = np.clip(vals, 0.1, 1.0)  # keep all clusters active
        return {g: float(vals[i]) for i, g in enumerate(avg_scores.keys())}
    

    def find_nclasses(self, jsd_condensed):
        from scipy.cluster.hierarchy import linkage, fcluster

        Z = linkage(jsd_condensed, method='average', optimal_ordering=True)

        best_k, sil_scores = self.evaluate_silhouette_scores(Z, jsd_condensed, k_min=2, k_max=11)

        return best_k, sil_scores


    def evaluate_silhouette_scores(self, Z, jsd_condensed, k_min=2, k_max=11):
        """
        """
        from scipy.cluster.hierarchy import fcluster
        from scipy.spatial.distance import squareform
        from sklearn.metrics import silhouette_score

        # Convert condensed distances to a full matrix for silhouette_score
        jsd_matrix = squareform(jsd_condensed)
        n_samples = jsd_matrix.shape[0]

        sil_scores = {}

        for k in range(k_min, min(k_max, n_samples - 1) + 1):
            clusters_k = fcluster(Z, k, criterion='maxclust')

            # Skip if clustering degenerates (e.g., all samples in one cluster)
            if len(np.unique(clusters_k)) < 2:
                continue

            score = silhouette_score(jsd_matrix, clusters_k, metric='precomputed')
            sil_scores[k] = score

        best_k = max(sil_scores, key=sil_scores.get)

        return best_k, sil_scores
    

    def _compute_group_distributions(self) -> dict[int, dict[int, float]]:
        """
        Compute normalized class distributions per cluster.
        Returns: {gid -> {class -> prob}}
        """
        group_dists = {}

        for gid, cids in self._g2c.items():
            agg = defaultdict(int)

            for cid in cids:
                client_dist = self._client_distributions.get(cid, {})
                for k, v in client_dist.items():
                    agg[int(k)] += int(v)

            total = sum(agg.values()) + 1e-8
            group_dists[gid] = {k: v / total for k, v in agg.items()}

        return group_dists


    def train(self, train_schedule: dict, max_local_round: int, threshold: float, patience: int,
              multithreading: bool = False) -> torch.nn.Module:
        
        logging.info("Starting FedHiKoD training.............")
        logging.info(f"Client dict: {self.client_dict}")

        self._client_distributions = {}
        for client in self.client_dict.values():
            self._client_distributions[client.client_id] = client.get_class_distribution()

        logging.info(f"Client distributions: {self._client_distributions}")

        client_ids, jsd_condensed = self._embed_for_clustering(self._client_distributions, total_classes=10, alpha=1)
        num_clusters, _ = self.find_nclasses(jsd_condensed)

        logging.info(f"Determined number of clusters: {num_clusters}")

        self._c2g, self._g2c = self.hierarchical_clustering(client_ids, jsd_condensed, num_clusters)
        self._group_distributions = self._compute_group_distributions()


        #self._c2g, self._g2c = self.hierarchical_clustering(client_ids, jsd_condensed, 10)

        for round in range(1, self.rounds + 1):
            logging.info(f"=== Global Round {round} ===")
            train_clients_ids = train_schedule.get(str(round), [])
            reports = {}

            train_clients = {cid: self.client_dict[cid] for cid in train_clients_ids}
            #----------------------------------------------------
            # Broadcast the global model to all clients
            #----------------------------------------------------

            self._broadcast(self.global_model, train_clients)

            num_data_points = {}
            for client in train_clients.values():
                
                leader_state = None
                #----------------------------------------------------
                # Send the leader model to the client if applicable
                #----------------------------------------------------
                if round > 1:
                    cid = client.client_id
                    gid = self._c2g[cid]

                    # Retrieve and send the leader model
                    if gid in inactive_clusters:
                        logging.info(f"Client {cid} in inactive cluster {gid}, skipping leader send.")
                    leader_state = self._leader_models_by_group.get(gid)
                    if leader_state is None:
                        logging.info(f"Not Sending leader model to client {cid} from group {gid}.")
                    #if leader_state is not None:
                        #client.receive_leader(leader_state)
                    if leader_state is not None:
                        client.receive_leader(
                            leader_weights=leader_state,
                            leader_dist=self._group_distributions[gid],
                            tau=tau_map[client.client_id],
                        )


                #----------------------------------------------------
                # Train the client with FedHiKoD local updates
                # receive trained model from client           
                # receive num of data points and class distribution 
                # ----------------------------------------------------
                client.train(round, 
                            max_local_round)
                
                num_data_points[client.client_id] = client.get_num_datapoints()
                self._client_distributions[client.client_id] = client.get_class_distribution()

                try:
                    score = self._collect_self_reports(client)
                    reports[client.client_id] = score
                except Exception as e:
                    logging.error(f"Client {client.client_id} reporting failed: {e}")
                logging.info(f"\n")

            inactive_clusters = [
                gid for gid, client_ids in self._g2c.items()
                if all(cid not in train_clients for cid in client_ids)
            ]

            logging.info(f"Inactive clusters: {inactive_clusters}")

            self._group_distributions = self._compute_group_distributions()

            # ----------------------------------------------------
            # Build group-aggregated leader models
            # ----------------------------------------------------

            self._leader_models_by_group = {}

            for group_id, client_ids in self._g2c.items():
                # Keep only clients that participated in this round
                valid_clients = [cid for cid in client_ids if cid in train_clients]
                if not valid_clients:
                    self._leader_models_by_group[group_id] = None
                    continue

                # Prepare list of models and weights for this group
                group_models = [train_clients[cid].get_model() for cid in valid_clients]
                group_weights_dict = {
                    cid: num_data_points[cid] / sum(num_data_points[c] for c in valid_clients)
                    for cid in valid_clients
                }
                group_weights =[group_weights_dict[cid] for cid in valid_clients]

                # Build the group leader
                logging.info("Subglobal (group:%s) model aggregation weights: %s", group_id, group_weights)
                group_leader = self._aggregate(group_models, group_weights)

                # Store as CPU state_dict (for later KD)
                self._leader_models_by_group[group_id] = {
                    k: v.detach().to('cpu', copy=False) for k, v in group_leader.state_dict().items()
                }

            cluster_q = self._cluster_quality(reports, self._c2g)
            tau_map = self._compute_tau_per_client(cluster_q)

            logging.info(f"Cluster Quality: {cluster_q}")
            logging.info(f"Number of data points: {num_data_points}")

            weighted_size = {}
            for cid, n in num_data_points.items():
                gid = self._c2g.get(cid)
                q = cluster_q.get(gid, 1.0)
                weighted_size[cid] = n * q

            total_weight = sum(weighted_size.values())
            weights = {cid: weighted_size[cid] / total_weight for cid in weighted_size}
        
            cid_order = list(train_clients.keys())
            model_list = [train_clients[cid].get_model() for cid in cid_order]
            w_list = [weights[cid] for cid in cid_order]

            logging.info(f"Global model aggregation weights: {w_list}")
            self.global_model = self._aggregate(model_list, w_list)

            # No cluster_q needed anymore unless used elsewhere
            cluster_q = self._cluster_quality(reports, self._c2g)

            #weighted_size = {}
            #for cid, n in num_data_points.items():
            #    score = float(reports.get(cid, {}).get("score", 1.0))
            #    weighted_size[cid] = n * score

            #total_weight = sum(weighted_size.values())
            #weights = {cid: weighted_size[cid] / total_weight for cid in weighted_size}

            #cid_order = list(train_clients.keys())
            #model_list = [train_clients[cid].get_model() for cid in cid_order]
            #w_list = [weights[cid] for cid in cid_order]

            #logging.info(f"Global model aggregation weights: {w_list}")
            #self.global_model = self._aggregate(model_list, w_list)

            self.save_checkpt(self.global_model, f"{self.checkpoint_path}/checkpoints/ckpt_{round}.pt")

            self._prev_cluster_q = getattr(self, "_prev_cluster_q", cluster_q)
            cluster_q = {
                g: 0.7 * self._prev_cluster_q.get(g, q) + 0.3 * q
                for g, q in cluster_q.items()
            }
            self._prev_cluster_q = cluster_q

        return self.global_model
    

class FedHiKoDClient(Client):
    """
    FedHiKoD-specific client that extends the base Client class
    with privacy-preserving fitness reporting and peer-committee auditing.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_classes = self.train_dataset.num_classes()
        self._leader_state = None  # holds the current group's leader weights
        self._p_i = None   # tensor of shape [num_classes]
        self._tau = 0.0    # scalar from server
        self._init_class_prior()



    def _init_class_prior(self):
        labels = [y for _, y in self.train_dataset]
        labels = np.asarray(labels)

        unique, counts = np.unique(labels, return_counts=True)
        K = 10 #self.num_classes   # define this appropriately

        p = np.zeros(K, dtype=float)
        for k, v in zip(unique, counts):
            p[int(k)] = v

        # Laplace smoothing
        p = (p + 1.0) / (p.sum() + K)
        self._p_i = torch.tensor(p, device=self.device, dtype=torch.float32)

    @staticmethod
    def _kd_loss_per_sample(student_logits: torch.Tensor,
                            teacher_logits: torch.Tensor,
                            temperature: float) -> torch.Tensor:
        T = temperature

        s_log = student_logits / (student_logits.std(dim=1, keepdim=True) + 1e-6)
        t_log = teacher_logits / (teacher_logits.std(dim=1, keepdim=True) + 1e-6)

        log_p_s = F.log_softmax(s_log / T, dim=1)   # (B, C)
        p_t = F.softmax(t_log / T, dim=1)           # (B, C)

        # kl_div with reduction="none": returns (B, C)
        kl = F.kl_div(log_p_s, p_t, reduction="none")
        # sum over classes → (B,)
        kl = kl.sum(dim=1)
        return kl * (T * T)


    def classwise_kd_weight(self, y: torch.Tensor,
                        lam_min: float = 0.1,
                        lam_max: float = 0.9) -> torch.Tensor:
        """
        y: (B,) int64 labels
        returns lambda_y: (B,) in [lam_min, lam_max]
        """
        p = self._p_i                         # (K,)
        p_y = p[y]                            # (B,)

        p_min = torch.min(p)
        p_max = torch.max(p)
        denom = (p_max - p_min + 1e-8)

        s = (p_max - p_y) / denom             # rare -> closer to 1
        lambda_y = lam_min + (lam_max - lam_min) * s
        return lambda_y


    def get_class_distribution(self) -> Dict[int, int]:
        """
        Get the class distribution in the training dataset.
        Assumes classification with integer class labels.
        """
        labels = [labels for _, labels in self.train_dataset]

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        else:
            labels = np.array(labels)

        unique, counts = np.unique(labels, return_counts=True)
        class_counts = {int(k): int(v) for k, v in zip(unique, counts)}
        return class_counts

    
    def report_fitness(self, recipe_hash, clip_range=( -0.9, 0.9)) -> Dict[str, object]:
        """
        """
        pre_loss, _  = self.evaluate(broadcast_model=True)   # evaluate before local train in this round
        post_loss, _ = self.evaluate(broadcast_model=False)  # evaluate after local train
        delta_loss = post_loss - pre_loss  # negative is good

        # 2) Clip + DP noise
        a, b = clip_range
        delta_loss = float(np.clip(delta_loss, a, b))

        # 4) Penalty for tiny validation sets
        n_eval = len(self.valdl.dataset)
        progress_term = -delta_loss

        logging.info(f"Progress term: {progress_term}")

        return {
            "client_id": self.client_id,
            "recipe_hash": recipe_hash,
            "n_eval": int(n_eval),
            "score": float(progress_term),
        }


    def receive_leader(self, leader_weights: dict | None,
                   leader_dist: dict | None,
                   tau: float = 0.0) -> None:
        self._leader_state = None
        self._leader_dist = leader_dist
        self._tau = float(tau)

        if leader_weights is None:
            return

        self._leader_state = {
            k: (v if torch.is_tensor(v) else torch.as_tensor(v)).detach().clone().to(self.device)
            for k, v in leader_weights.items()
        }



    def _build_teacher_from_leader(self) -> torch.nn.Module | None:
        """
        Build a frozen teacher model from self._leader_state.
        Returns None if no leader state is available.
        """
        leader_state = self._leader_state
        if leader_state is None:
            return None

        # Create a copy of the same architecture
        teacher = copy.deepcopy(self.local_model).to(self.device)

        # The leader weights you stored were only parameter tensors (no buffers).
        # strict=False lets us load what's available and ignore missing buffers (e.g., BN running stats).
        try:
            teacher.load_state_dict(leader_state, strict=False)
        except Exception as e:
            logging.error(f"[{self.client_id}] Failed to load leader state into teacher: {e}")
            return None

        # Freeze teacher
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher
        

    def adaptive_temperature(self, round_idx: int,
                         total_rounds: int,
                         T_min: float = 1.0,
                         T_max: float = 8.0,
                         mode: str = "linear") -> float:
        ratio = round_idx / max(total_rounds, 1)
        if mode == "linear":
            return T_max - (T_max - T_min) * ratio
        elif mode == "exp":
            return T_min + (T_max - T_min) * math.exp(-3 * ratio)
        else:
            return T_max
        

    @staticmethod
    def _kd_loss(student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                temperature: float) -> torch.Tensor:
        T = temperature

        # normalize logits to prevent scale collapse
        s_log = student_logits / (student_logits.std(dim=1, keepdim=True) + 1e-6)
        t_log = teacher_logits / (teacher_logits.std(dim=1, keepdim=True) + 1e-6)

        log_p_s = F.log_softmax(s_log / T, dim=1)
        p_t = F.softmax(t_log / T, dim=1)

        kd = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
        return kd
    

    def train(
    self,
    global_round: int,
    max_local_round: int,
    grad_clip: float = 1.0,
    use_kd: bool = True,
) -> torch.nn.Module:

        # snapshot the global model
        self.global_model = copy.deepcopy(self.local_model).to(self.device)
        self.local_model.train()

        # ----------------------------
        # Build teacher model
        # ----------------------------
        teacher_leader = self._build_teacher_from_leader() if use_kd else None
        if teacher_leader is None:
            use_kd = False
            logging.info(f"[{self.client_id}] KD disabled (no valid teacher).")

        # ----------------------------
        # Adaptive temperature
        # ----------------------------
        kd_T = self.adaptive_temperature(
            round_idx=global_round,
            total_rounds=300,
            T_min=1.0,
            T_max=8.0,
            mode="linear"
        ) if use_kd else 1.0

        # ----------------------------
        # Training loop
        # ----------------------------
        for epoch in range(max_local_round):
            batch_loss = []

            for batch_idx, (x, y) in enumerate(self.traindl):
                x, y = x.to(self.device), y.to(self.device)

                # ----------------------------
                # Forward pass (student)
                # ----------------------------
                student_logits = self.local_model(x)

                # ----- hard loss per sample -----
                if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
                    if y.ndim > 1:
                        if y.size(-1) == 1:
                            y = y.squeeze(-1)
                        else:
                            y = y.argmax(dim=-1)
                    y = y.long()

                hard_loss_vec = F.cross_entropy(
                    student_logits, y, reduction="none"
                )  # (B,)

                # ----------------------------
                # KD loss per sample
                # ----------------------------
                if use_kd:
                    with torch.no_grad():
                        leader_logits = teacher_leader(x)

                    kd_loss_vec = self._kd_loss_per_sample(
                        student_logits,
                        leader_logits,
                        temperature=kd_T
                    )  # (B,)

                    # class-wise KD weight λᵢ(y)
                    lambda_y = self.classwise_kd_weight(y)     # (B,)

                    # cluster trust τᵢ
                    tau_i = torch.tensor(
                        self._tau, device=self.device, dtype=torch.float32
                    )

                    # effective KD authority αᵢ(y)
                    alpha_y = tau_i * lambda_y                 # (B,)

                    # final mixed loss
                    loss_vec = (1.0 - alpha_y) * hard_loss_vec \
                            + alpha_y * kd_loss_vec

                    loss = loss_vec.mean()

                else:
                    loss = hard_loss_vec.mean()

                # ----------------------------
                # Backpropagation
                # ----------------------------
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.local_model.parameters(), grad_clip
                )
                self.optimizer.step()

                batch_loss.append(loss.item())

            avg_loss = float(sum(batch_loss) / max(1, len(batch_loss)))
            logging.info(
                f"Client: {self.client_id:<10} "
                f"Epoch: {epoch + 1:<2} "
                f"Avg Loss: {avg_loss:<10.6f} "
                f"Global Round: {global_round} "
                f"Tau: {self._tau:.4f}"
            )

        # clear leader after training
        self._leader_state = None

        return self.local_model
    


class FedHiKoDServer(Server):

    """
    The federated learning server class for FedHiKoD.
    """

    def __init__(self, *args, fitness_cfg: FitnessCfg = FitnessCfg(), **kwargs):
        super().__init__(*args, **kwargs)
        self.fitness_cfg = fitness_cfg

        self._g_ema = None             # torch.Tensor on CPU or None
        self._g_beta = 0.5             # momentum factor for g_EMA
        self._g_num_final_layers = 2   # must match clients' alignment layers
        self._groups: dict[str, int] = {}              # cid -> group id (fixed after round 1)
        self._client_distributions: dict[str, dict[int, float]] = {}
        self._leader_models_by_group: dict[int, dict] = {}  # gid -> state_dict
        self._client_trust = defaultdict(lambda: 0.0)  # initialize to 0

    
    def _client_weight_multiplier(self, train_clients_ids, kappa=0.7):
        # kappa controls how strong trust can bend weights (0.5 => 0.5–1.5)
        trusts = np.array([self._client_trust[cid] for cid in train_clients_ids], dtype=float)

        # Normalize to zero-mean, unit variance
        mean = trusts.mean()
        std = trusts.std() + 1e-8
        z = (trusts - mean) / std

        # Squash with tanh so outliers can’t explode
        # z in ~[-2, 2] => tanh(z) in ~[-0.96, 0.96]
        s = np.tanh(z)

        multipliers = {}
        for cid, s_c in zip(train_clients_ids, s):
            # map s_c ∈ [-1,1] to [1 - kappa, 1 + kappa]
            m = 1.0 + kappa * s_c
            multipliers[cid] = float(np.clip(m, 1.0 - kappa, 1.0 + kappa))
        return multipliers

  
    def _update_client_trust(self, reports, beta=0.7):
        # beta: how much weight to give past trust (0.8 = slow changes)
        for cid, rpt in reports.items():
            prog = float(rpt.get("score", 0.0))
            old = self._client_trust[cid]
            new = beta * old + (1 - beta) * prog
            self._client_trust[cid] = new


    def flatten_last_modules(self, model: torch.nn.Module, last_n_modules: int | None = None) -> torch.Tensor:
        """
        Flatten parameters from the last `last_n_modules` child modules of `model`.
        If `last_n_modules` is None, flatten all parameters.
        """
        with torch.no_grad():
            # children keep registration order: fc1, relu, fc2
            modules = [m for _, m in model.named_children()]
            # drop non-param modules like ReLU
            modules = [m for m in modules if any(p.requires_grad for p in m.parameters(recurse=False))]

            if last_n_modules is None:
                selected = modules
            else:
                selected = modules[-last_n_modules:]

            flats = []
            for m in selected:
                for p in m.parameters(recurse=False):
                    flats.append(p.detach().cpu().flatten())
            if not flats:  # fallback to all params if something odd happens
                flats = [p.detach().cpu().flatten() for p in model.parameters()]
            return torch.cat(flats)


    def _global_update_vector(self, old_model: torch.nn.Module, new_model: torch.nn.Module,
                              num_final_layers: int | None = 2) -> torch.Tensor:
        with torch.no_grad():
            v_old = self.flatten_last_modules(old_model)
            v_new = self.flatten_last_modules(new_model)
            dtheta = (v_new - v_old).detach().cpu()
        return dtheta
    

    def _update_global_momentum(self, dtheta: torch.Tensor) -> torch.Tensor:
        # no-op if the update is degenerate
        if dtheta is None or torch.norm(dtheta) == 0:
            return self._g_ema

        if self._g_ema is None or torch.numel(self._g_ema) != torch.numel(dtheta):
            # initialize on first use or if shape changed
            self._g_ema = dtheta.clone().detach().cpu()
        else:
            self._g_ema = (self._g_beta * self._g_ema) + ((1.0 - self._g_beta) * dtheta.detach().cpu())
        return self._g_ema


    def _embed_for_clustering(self, distributions: Dict, total_classes: int, alpha: int = 1)-> Tuple[list, np.ndarray]:
        """
        Create a fixed-size embedding vector from each client's class distribution
        and compute pairwise Jensen-Shannon distances.
        """
        from scipy.spatial.distance import pdist
        from scipy.spatial.distance import jensenshannon

        client_ids = list(distributions.keys())
        vectors = []

        for cid in client_ids:
            counts = np.array([distributions[cid].get(i, 0) for i in range(total_classes)], dtype=float)
            counts += alpha  # Laplace smoothing
            counts /= counts.sum()
            vectors.append(counts)

        vectors = np.vstack(vectors)
        jsd_condensed = pdist(vectors, metric=lambda u, v: jensenshannon(u, v))
        return client_ids, jsd_condensed


    def hierarchical_clustering(self, client_ids, jsd_condensed, num_clusters):
        """
        Perform hierarchical clustering using Jensen–Shannon distances.
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from collections import defaultdict

        # Perform linkage using precomputed distances
        Z = linkage(jsd_condensed, method='average', optimal_ordering=True)
        clusters = fcluster(Z, t=num_clusters, criterion='maxclust')

        client_to_cluster = {}
        cluster_to_clients = defaultdict(list)

        for cid, cluster_id in zip(client_ids, clusters):
            client_to_cluster[cid] = cluster_id
            cluster_to_clients[cluster_id].append(cid)

        return client_to_cluster, cluster_to_clients


    def _collect_self_reports(self, client: Client) -> dict[str, dict]:
        cfg = self.fitness_cfg
        recipe_hash = hashlib.sha256(f"metric:loss|clip:{cfg.clip_range}".encode()).hexdigest()

        if not hasattr(client, "report_fitness"):
            raise RuntimeError(f"Client {client.client_id} lacks report_fitness()")
        
        # these are hyperparams
        rpt = client.report_fitness(
            recipe_hash=recipe_hash,
            clip_range=cfg.clip_range,
        )

        return rpt

 
    def _aggregate(self, trained_clients, weights):
        """
        Aggregate the models of the clients using FedHiKoD aggregation strategy.
        """
        return weighted_avg(self.global_model, trained_clients, weights)


    def _cluster_quality(self, reports: dict[str, dict], c2g: dict[str, int]) -> dict[int, float]:
        """
        Aggregate client 'score' values into per-cluster quality.
        Normalizes to [0.1, 1.0] to avoid zero weights.
        """

        cluster_scores = defaultdict(list)
        for cid, rpt in reports.items():
            gid = c2g.get(cid)
            score = float(rpt.get("score", 0.0))
            cluster_scores[gid].append(score)

        if not cluster_scores:
            return {}

        avg_scores = {g: np.mean(v) for g, v in cluster_scores.items()}
        vals = np.array(list(avg_scores.values()))
        vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        vals = np.clip(vals, 0.1, 1.0)  # keep all clusters active
        return {g: float(vals[i]) for i, g in enumerate(avg_scores.keys())}
    

    def find_nclasses(self, jsd_condensed):
        from scipy.cluster.hierarchy import linkage, fcluster

        Z = linkage(jsd_condensed, method='average', optimal_ordering=True)

        best_k, sil_scores = self.evaluate_silhouette_scores(Z, jsd_condensed, k_min=2, k_max=11)

        return best_k, sil_scores


    def evaluate_silhouette_scores(self, Z, jsd_condensed, k_min=2, k_max=11):
        """
        """
        from scipy.cluster.hierarchy import fcluster
        from scipy.spatial.distance import squareform
        from sklearn.metrics import silhouette_score

        # Convert condensed distances to a full matrix for silhouette_score
        jsd_matrix = squareform(jsd_condensed)
        n_samples = jsd_matrix.shape[0]

        sil_scores = {}

        for k in range(k_min, min(k_max, n_samples - 1) + 1):
            clusters_k = fcluster(Z, k, criterion='maxclust')

            # Skip if clustering degenerates (e.g., all samples in one cluster)
            if len(np.unique(clusters_k)) < 2:
                continue

            score = silhouette_score(jsd_matrix, clusters_k, metric='precomputed')
            sil_scores[k] = score

        best_k = max(sil_scores, key=sil_scores.get)

        return best_k, sil_scores
    
    def _fairness_multipliers(self, fair_need: dict[str, float], kappa: float = 0.5):
        cids = list(fair_need.keys())
        vals = np.array([fair_need[c] for c in cids], dtype=float)

        if np.all(vals == 0):
            # everyone equal → return 1.0 multipliers
            return {cid: 1.0 for cid in cids}

        mean = vals.mean()
        std = vals.std() + 1e-8
        z = (vals - mean) / std
        s = np.tanh(z)   # [-1,1]

        mult = {}
        for cid, s_c in zip(cids, s):
            # here: positive z (high need) → s_c > 0 → m > 1
            m = 1.0 + kappa * s_c
            mult[cid] = float(np.clip(m, 1.0 - kappa, 1.0 + kappa))
        return mult

    def train(self, train_schedule: dict, max_local_round: int, threshold: float, patience: int,
              multithreading: bool = False) -> torch.nn.Module:
        
        logging.info("Starting FedHiKoD training.............")
        logging.info(f"Client dict: {self.client_dict}")

        self._client_distributions = {}
        for client in self.client_dict.values():
            self._client_distributions[client.client_id] = client.get_class_distribution()

        logging.info(f"Client distributions: {self._client_distributions}")

        client_ids, jsd_condensed = self._embed_for_clustering(self._client_distributions, total_classes=10, alpha=1)
        num_clusters, _ = self.find_nclasses(jsd_condensed)

        logging.info(f"Determined number of clusters: {num_clusters}")

        self._c2g, self._g2c = self.hierarchical_clustering(client_ids, jsd_condensed, num_clusters)

        for r in range(1, self.rounds + 1):
            logging.info(f"=== Global Round {r} ===")
            train_clients_ids = train_schedule.get(str(r), [])
            reports = {}

            train_clients = {cid: self.client_dict[cid] for cid in train_clients_ids}
            #----------------------------------------------------
            # Broadcast the global model to all clients
            #----------------------------------------------------

            self._broadcast(self.global_model, train_clients)

            num_data_points = {}
            for client in train_clients.values():
                
                leader_state = None
                #----------------------------------------------------
                # Send the leader model to the client if applicable
                #----------------------------------------------------
                if r > 10:
                    cid = client.client_id
                    gid = self._c2g[cid]

                    # Retrieve and send the leader model
                    if gid in inactive_clusters:
                        logging.info(f"Client {cid} in inactive cluster {gid}, skipping leader send.")
                    leader_state = self._leader_models_by_group.get(gid)
                    if leader_state is None:
                        logging.info(f"Not Sending leader model to client {cid} from group {gid}.")
                    if leader_state is not None:
                        client.receive_leader(leader_state)

                #----------------------------------------------------
                # Train the client with FedHiKoD local updates
                # receive trained model from client           
                # receive num of data points and class distribution 
                # ----------------------------------------------------
                client.train(r, 
                            max_local_round)
                
                num_data_points[client.client_id] = client.get_num_datapoints()
                self._client_distributions[client.client_id] = client.get_class_distribution()

                try:
                    score = self._collect_self_reports(client)
                    reports[client.client_id] = score
                except Exception as e:
                    logging.error(f"Client {client.client_id} reporting failed: {e}")
                logging.info(f"\n")


            inactive_clusters = [
                gid for gid, client_ids in self._g2c.items()
                if all(cid not in train_clients for cid in client_ids)
            ]
            logging.info(f"Inactive clusters: {inactive_clusters}")

            # ----------------------------------------------------
            # Build group-aggregated leader models
            # ----------------------------------------------------
            self._leader_models_by_group = {}
            for group_id, client_ids in self._g2c.items():
                valid_clients = [cid for cid in client_ids if cid in train_clients]
                if not valid_clients:
                    self._leader_models_by_group[group_id] = None
                    continue

                group_models = [train_clients[cid].get_model() for cid in valid_clients]
                group_weights_dict = {
                    cid: num_data_points[cid] / sum(num_data_points[c] for c in valid_clients)
                    for cid in valid_clients
                }
                group_weights = [group_weights_dict[cid] for cid in valid_clients]

                logging.info("Subglobal (group:%s) model aggregation weights: %s", group_id, group_weights)
                group_leader = self._aggregate(group_models, group_weights)

                self._leader_models_by_group[group_id] = {
                    k: v.detach().to('cpu', copy=False) for k, v in group_leader.state_dict().items()
                }

            # ----------------------------------------------------
            # Cluster quality + trust update
            # ----------------------------------------------------
            cluster_q = self._cluster_quality(reports, self._c2g)
            logging.info(f"Cluster Quality: {cluster_q}")
            logging.info(f"Number of data points: {num_data_points}")

            # 1) update trust from this round's reports (ONCE)
            self._update_client_trust(reports)

            # 2) smooth cluster quality
            self._prev_cluster_q = getattr(self, "_prev_cluster_q", cluster_q)
            cluster_q = {
                g: 0.7 * self._prev_cluster_q.get(g, q) + 0.3 * q
                for g, q in cluster_q.items()
            }
            self._prev_cluster_q = cluster_q

            # 3) build hierarchical trust = client_trust × cluster_q
            effective_trust = {}
            for cid in num_data_points.keys():
                gid = self._c2g[cid]
                gq = cluster_q.get(gid, 1.0)
                effective_trust[cid] = float(self._client_trust[cid] * gq)

            trust_backup = self._client_trust
            self._client_trust = defaultdict(lambda: 0.0, effective_trust)

            mult = self._client_weight_multiplier(list(num_data_points.keys()), kappa=0.5)

            self._client_trust = trust_backup

            logging.info(f"Trust values: {[ (cid, self._client_trust[cid]) for cid in num_data_points.keys()]}")
            logging.info(f"Hierarchical multipliers: {mult}")

            accs = {cid: float(rpt["post_f1"]) for cid, rpt in reports.items()}
            mean_acc = sum(accs.values()) / max(len(accs), 1)

            # fairness priority: how far BELOW the mean this client is
            fair_need = {
                cid: max(mean_acc - accs[cid], 0.0)  # >0 if below avg, 0 if at/above avg
                for cid in accs.keys()
            }

            fair_mult = self._fairness_multipliers(fair_need, kappa=0.5)
            logging.info(f"Fairness multipliers (cid → need → mult): %s",
                        {cid: (round(fair_need[cid], 4), round(fair_mult[cid], 3)) for cid in fair_mult})

            # 4) Base FedAvg (or tempered) weights
            total_n = sum(num_data_points.values())
            base_w = {cid: n / total_n for cid, n in num_data_points.items()}  # or use n**tau

            logging.info(f"Base weights: {base_w}")

            # 5) Final weights = base * multiplier
            raw = {cid: base_w[cid] * mult[cid] for cid in num_data_points.keys()}
            Z = sum(raw.values()) + 1e-12
            weights = {cid: v / Z for cid, v in raw.items()}

            # 1) trust-based multipliers (what you already have)
            trust_mult = self._client_weight_multiplier(list(num_data_points.keys()), kappa=0.5)

            # 2) fairness multipliers based on accuracy gap
            fair_mult = self._fairness_multipliers(fair_need, kappa=0.5)

            # 3) combine (geometric or convex)
            alpha = 0  # 0→no fairness, 1→only fairness
            combined_mult = {
                cid: (trust_mult[cid] ** (1 - alpha)) * (fair_mult[cid] ** alpha)
                for cid in num_data_points.keys()
            }

            raw = {cid: base_w[cid] * combined_mult[cid] for cid in num_data_points.keys()}
            Z = sum(raw.values()) + 1e-12
            weights = {cid: v / Z for cid, v in raw.items()}

            cid_order = list(train_clients.keys())
            model_list = [train_clients[cid].get_model() for cid in cid_order]
            w_list = [weights[cid] for cid in cid_order]

            logging.info(f"Global model aggregation weights: {w_list}")

            logging.info(
            "Final weights (cid → base → final): %s",
            {
                cid: (round(base_w[cid], 6), round(weights[cid], 6))
                for cid in cid_order
            }
            )
            self.global_model = self._aggregate(model_list, w_list)

            self.save_checkpt(self.global_model, f"{self.checkpoint_path}/checkpoints/ckpt_{r}.pt")

        return self.global_model
        


"""
            inactive_clusters = [
                gid for gid, client_ids in self._g2c.items()
                if all(cid not in train_clients for cid in client_ids)
            ]

            logging.info(f"Inactive clusters: {inactive_clusters}")

            # ----------------------------------------------------
            # Build group-aggregated leader models
            # ----------------------------------------------------

            self._leader_models_by_group = {}

            for group_id, client_ids in self._g2c.items():
                # Keep only clients that participated in this round
                valid_clients = [cid for cid in client_ids if cid in train_clients]
                if not valid_clients:
                    self._leader_models_by_group[group_id] = None
                    continue

                # Prepare list of models and weights for this group
                group_models = [train_clients[cid].get_model() for cid in valid_clients]
                group_weights_dict = {
                    cid: num_data_points[cid] / sum(num_data_points[c] for c in valid_clients)
                    for cid in valid_clients
                }
                group_weights =[group_weights_dict[cid] for cid in valid_clients]

                # Build the group leader
                logging.info("Subglobal (group:%s) model aggregation weights: %s", group_id, group_weights)
                group_leader = self._aggregate(group_models, group_weights)

                # Store as CPU state_dict (for later KD)
                self._leader_models_by_group[group_id] = {
                    k: v.detach().to('cpu', copy=False) for k, v in group_leader.state_dict().items()
                }

            cluster_q = self._cluster_quality(reports, self._c2g)
            logging.info(f"Cluster Quality: {cluster_q}")
            logging.info(f"Number of data points: {num_data_points}")

            self._update_client_trust(reports)

            # smooth cluster quality
            self._prev_cluster_q = getattr(self, "_prev_cluster_q", cluster_q)
            cluster_q = {
                g: 0.7 * self._prev_cluster_q.get(g, q) + 0.3 * q
                for g, q in cluster_q.items()
            }
            self._prev_cluster_q = cluster_q

            # build hierarchical trust
            effective_trust = {}
            for cid in num_data_points.keys():
                gid = self._c2g[cid]
                gq = cluster_q.get(gid, 1.0)
                effective_trust[cid] = float(self._client_trust[cid] * gq)

            trust_backup = self._client_trust
            self._client_trust = defaultdict(lambda: 0.0, effective_trust)

            mult = self._client_weight_multiplier(list(num_data_points.keys()), kappa=0.5)

            self._client_trust = trust_backup

            #weighted_size = {}
            #for cid, n in num_data_points.items():
            #    gid = self._c2g.get(cid)
            #    q = cluster_q.get(gid, 1.0)
            #    weighted_size[cid] = n * q

            #total_weight = sum(weighted_size.values())
            #weights = {cid: weighted_size[cid] / total_weight for cid in weighted_size}

            # 1) Base FedAvg weights
            total_n = sum(num_data_points.values())
            base_w = {cid: n / total_n for cid, n in num_data_points.items()}

            # 2) Update trust from this round's reports
            self._update_client_trust(reports)

            # 3) Get bounded multipliers
            mult = self._client_weight_multiplier(list(num_data_points.keys()), kappa=0.5)

            logging.info(f"Trust values: {[ (cid, self._client_trust[cid]) for cid in num_data_points.keys()]}")
            logging.info(f"Trust multipliers: {mult}")

            # 4) Final weights = FedAvg * multiplier
            raw = {cid: base_w[cid] * mult[cid] for cid in num_data_points.keys()}
            Z = sum(raw.values()) + 1e-12
            weights = {cid: v / Z for cid, v in raw.items()}

        
            cid_order = list(train_clients.keys())
            model_list = [train_clients[cid].get_model() for cid in cid_order]
            w_list = [weights[cid] for cid in cid_order]

            logging.info(f"Global model aggregation weights: {w_list}")
            self.global_model = self._aggregate(model_list, w_list)

            # No cluster_q needed anymore unless used elsewhere
            #cluster_q = self._cluster_quality(reports, self._c2g)

            #weighted_size = {}
            #for cid, n in num_data_points.items():
            #    score = float(reports.get(cid, {}).get("score", 1.0))
            #    weighted_size[cid] = n * score

            #total_weight = sum(weighted_size.values())
            #weights = {cid: weighted_size[cid] / total_weight for cid in weighted_size}

            #cid_order = list(train_clients.keys())
            #model_list = [train_clients[cid].get_model() for cid in cid_order]
            #w_list = [weights[cid] for cid in cid_order]

            #logging.info(f"Global model aggregation weights: {w_list}")
            #self.global_model = self._aggregate(model_list, w_list)

            self.save_checkpt(self.global_model, f"{self.checkpoint_path}/checkpoints/ckpt_{round}.pt")

            self._prev_cluster_q = getattr(self, "_prev_cluster_q", cluster_q)
            cluster_q = {
                g: 0.7 * self._prev_cluster_q.get(g, q) + 0.3 * q
                for g, q in cluster_q.items()
            }
            self._prev_cluster_q = cluster_q

        return self.global_model

"""

"""
    def train(self, train_schedule: dict, max_local_round: int, threshold: float, patience: int,
              multithreading: bool = False) -> torch.nn.Module:
        
        logging.info("Starting FedHiKoD training.............")
        logging.info(f"Client dict: {self.client_dict}")

        self._client_distributions = {}
        #for client in self.client_dict.values():
        #    self._client_distributions[client.client_id] = client.get_class_distribution()

        #logging.info(f"Client distributions: {self._client_distributions}")

        #client_ids, jsd_condensed = self._embed_for_clustering(self._client_distributions, total_classes=10, alpha=1)
        #self._c2g, self._g2c = self.hierarchical_clustering(client_ids, jsd_condensed, 10)

        # Get the client IDs that will participate in round 1
        round1_client_ids = set(train_schedule.get(str(1), []))
        for client in self.client_dict.values():
            if client.client_id in round1_client_ids:
                self._client_distributions[client.client_id] = client.get_class_distribution()

        logging.info(f"Client distributions (Round 1 only): {self._client_distributions}")

        client_ids, jsd_condensed = self._embed_for_clustering(self._client_distributions, total_classes=10, alpha=1)
        #num_clusters, _ = self.find_nclasses(jsd_condensed)

        self._c2g, self._g2c = self.hierarchical_clustering(client_ids, jsd_condensed, 10)

        for round in range(1, self.rounds + 1):
            logging.info(f"=== Global Round {round} ===")
            train_clients_ids = train_schedule.get(str(1), [])
            reports = {}

            train_clients = {cid: self.client_dict[cid] for cid in train_clients_ids}
            #----------------------------------------------------
            # Broadcast the global model to all clients
            #----------------------------------------------------

            self._broadcast(self.global_model, train_clients)

            num_data_points = {}
            for client in train_clients.values():
                
                leader_state = None
                #----------------------------------------------------
                # Send the leader model to the client if applicable
                #----------------------------------------------------
                if round > 1:
                    cid = client.client_id
                    gid = self._c2g[cid]

                    # Retrieve and send the leader model
                    if gid in inactive_clusters:
                        logging.info(f"Client {cid} in inactive cluster {gid}, skipping leader send.")
                    leader_state = self._leader_models_by_group.get(gid)
                    if leader_state is None:
                        logging.info(f"Not Sending leader model to client {cid} from group {gid}.")
                    if leader_state is not None:
                        client.receive_leader(leader_state)

                #----------------------------------------------------
                # Train the client with FedHiKoD local updates
                # receive trained model from client           
                # receive num of data points and class distribution 
                # ----------------------------------------------------
                client.train(round, 
                            max_local_round)
                
                num_data_points[client.client_id] = client.get_num_datapoints()
                self._client_distributions[client.client_id] = client.get_class_distribution()

                try:
                    score = self._collect_self_reports(client)
                    reports[client.client_id] = score
                except Exception as e:
                    logging.error(f"Client {client.client_id} reporting failed: {e}")
                logging.info(f"\n")

            inactive_clusters = [
                gid for gid, client_ids in self._g2c.items()
                if all(cid not in train_clients for cid in client_ids)
            ]

            logging.info(f"Inactive clusters: {inactive_clusters}")

            # ----------------------------------------------------
            # Build group-aggregated leader models
            # ----------------------------------------------------

            self._leader_models_by_group = {}

            for group_id, client_ids in self._g2c.items():
                # Keep only clients that participated in this round
                valid_clients = [cid for cid in client_ids if cid in train_clients]
                if not valid_clients:
                    self._leader_models_by_group[group_id] = None
                    continue

                # Prepare list of models and weights for this group
                group_models = [train_clients[cid].get_model() for cid in valid_clients]
                group_weights_dict = {
                    cid: num_data_points[cid] / sum(num_data_points[c] for c in valid_clients)
                    for cid in valid_clients
                }
                group_weights =[group_weights_dict[cid] for cid in valid_clients]

                # Build the group leader
                logging.info("Subglobal (group:%s) model aggregation weights: %s", group_id, group_weights)
                group_leader = self._aggregate(group_models, group_weights)

                # Store as CPU state_dict (for later KD)
                self._leader_models_by_group[group_id] = {
                    k: v.detach().to('cpu', copy=False) for k, v in group_leader.state_dict().items()
                }

            cluster_q = self._cluster_quality(reports, self._c2g)
            logging.info(f"Cluster Quality: {cluster_q}")
            logging.info(f"Number of data points: {num_data_points}")

            weighted_size = {}
            for cid, n in num_data_points.items():
                gid = self._c2g.get(cid)
                q = cluster_q.get(gid, 1.0)
                weighted_size[cid] = n * q

            total_weight = sum(weighted_size.values())
            weights = {cid: weighted_size[cid] / total_weight for cid in weighted_size}
        
            cid_order = list(train_clients.keys())
            model_list = [train_clients[cid].get_model() for cid in cid_order]
            w_list = [weights[cid] for cid in cid_order]

            logging.info(f"Global model aggregation weights: {w_list}")
            self.global_model = self._aggregate(model_list, w_list)

            # No cluster_q needed anymore unless used elsewhere
            cluster_q = self._cluster_quality(reports, self._c2g)

            #weighted_size = {}
            #for cid, n in num_data_points.items():
            #    score = float(reports.get(cid, {}).get("score", 1.0))
            #    weighted_size[cid] = n * score

            #total_weight = sum(weighted_size.values())
            #weights = {cid: weighted_size[cid] / total_weight for cid in weighted_size}

            #cid_order = list(train_clients.keys())
            #model_list = [train_clients[cid].get_model() for cid in cid_order]
            #w_list = [weights[cid] for cid in cid_order]

            #logging.info(f"Global model aggregation weights: {w_list}")
            #self.global_model = self._aggregate(model_list, w_list)

            self.save_checkpt(self.global_model, f"{self.checkpoint_path}/checkpoints/ckpt_{round}.pt")

            self._prev_cluster_q = getattr(self, "_prev_cluster_q", cluster_q)
            cluster_q = {
                g: 0.7 * self._prev_cluster_q.get(g, q) + 0.3 * q
                for g, q in cluster_q.items()
            }
            self._prev_cluster_q = cluster_q

        return self.global_model
"""
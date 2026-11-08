import torch
import numpy as np
from models.backbone import clear_adj_cache

class ContinualTrainer:
    """
    Class-Incremental Continual Graph Learning Trainer for TriP.
    Evaluates under official CGLB protocol:
      - At stage t, evaluate on test sets of all tasks j <= t
      - Prediction candidate space is ALL classes seen so far: C_seen = U_{j<=t} C_j
      - Tracks accuracy matrix M in R^{T x T}
    """
    def __init__(self, model, tasks, lr=0.005, epochs=60, weight_decay=5e-4):
        self.model = model
        self.tasks = tasks
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.weight_decay = float(weight_decay)
        self.num_tasks = len(tasks)
        self.accuracy_matrix = np.zeros((self.num_tasks, self.num_tasks))

    def train_continual(self, x, edge_index, y):
        observed_classes = []

        for t, task in enumerate(self.tasks):
            current_classes = task["classes"]
            previous_classes = list(observed_classes)
            observed_classes.extend(current_classes)

            train_nodes = task["train_nodes"].to(self.model.device)
            print(f"\n==========================================")
            print(f"=== Continual Task {t + 1}/{self.num_tasks}: Classes {current_classes} ===")
            print(f"==========================================")

            # 1. Compute and register task prototype m_t (Eq. 3)
            self.model.task_prototype_buffer.register_task_prototype(t, x, train_nodes)

            # 2. Add lightweight Feature, Class, and Topological prompts for task t
            self.model.add_task_prompts(t)
            prompts = self.model.get_task_prompts(t)

            # 3. Setup optimizer exclusively for current task's prompt parameters
            prompt_params = []
            for p_mod in prompts:
                if p_mod is not None:
                    prompt_params.extend(list(p_mod.parameters()))
            optimizer = torch.optim.Adam(prompt_params, lr=self.lr, weight_decay=self.weight_decay)

            # Initial prototype computation before training
            with torch.no_grad():
                h_tilde_init = self.model.encode(x, edge_index, t)
                self.model.class_prototype_buffer.update_prototypes(
                    h_tilde_init, y, train_nodes, current_classes
                )

            # 4. Continual Prompt Optimization
            self.model.train()
            for epoch in range(1, self.epochs + 1):
                optimizer.zero_grad()

                # Forward pass through prompted GNN
                h_tilde = self.model.encode(x, edge_index, t)

                # Recompute current class prototypes at each epoch (Section 4.4.2)
                self.model.class_prototype_buffer.update_prototypes(
                    h_tilde, y, train_nodes, current_classes
                )

                # Compute Unified Prompt Template Loss
                loss = self.model.compute_unified_prompt_loss(
                    h_tilde, y, train_nodes, current_classes, previous_classes
                )

                loss.backward()
                optimizer.step()

                if epoch % 20 == 0 or epoch == self.epochs:
                    print(f"  [Task {t + 1} | Epoch {epoch:03d}/{self.epochs:03d}] Loss: {loss.item():.4f}")

            # Freeze current prompt parameters permanently
            for p in prompt_params:
                p.requires_grad = False
            del optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 5. Evaluate after learning task t across all seen tasks j <= t
            print(f"\n--- Evaluating after Task {t + 1} across all {len(observed_classes)} seen classes ---")
            for j in range(t + 1):
                test_nodes = self.tasks[j]["test_nodes"].to(self.model.device)
                test_y = y[test_nodes].to(self.model.device)

                preds = self.model.predict(x, edge_index, test_nodes, observed_classes)
                acc = (preds == test_y).float().mean().item() * 100.0
                self.accuracy_matrix[t, j] = acc
                print(f"  Accuracy on Task {j + 1} test set: {acc:.2f}%")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        clear_adj_cache()

        # 6. Compute Final Continual Metrics (Eq. 19)
        T = self.num_tasks
        avg_accuracy = np.mean([self.accuracy_matrix[T - 1, j] for j in range(T)])
        if T > 1:
            forgetting_list = [self.accuracy_matrix[T - 1, j] - self.accuracy_matrix[j, j] for j in range(T - 1)]
            avg_forgetting = np.mean(forgetting_list)
        else:
            avg_forgetting = 0.0

        print(f"\n==========================================")
        print(f"=== Continual Graph Learning Finished ===")
        print(f"Average Accuracy (AA): {avg_accuracy:.2f}%")
        print(f"Average Forgetting (AF): {avg_forgetting:.2f}%")
        print(f"==========================================")

        return {
            "accuracy_matrix": self.accuracy_matrix,
            "average_accuracy": avg_accuracy,
            "average_forgetting": avg_forgetting
        }

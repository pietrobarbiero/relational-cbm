import os
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset

from datasets.hanoi import hanoi_toy_dataset
from experiments.model import RelationalCBMDeep
from rcbm.logic.commons import Rule, Domain
from rcbm.logic.grounding import DomainGrounder
from rcbm.logic.indexing import DictBasedIndexer
from rcbm.logic.semantics import GodelTNorm

# disable cuda
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

pl.seed_everything(42)


def main():
    n_samples = 1000
    n_positions = 7
    n_disks = 3
    n_sizes = 10
    random_seed = 42
    learning_rate = 0.001
    batch_size = 1
    limit_batches = 1.0
    emb_size = 30
    manifold_arity = 3
    task_weight = 0.3
    fold = 1
    logic = GodelTNorm()

    body = [
        'top(X,Y)',
        'top(Y,X)',
        'top(X,Z)',
        'top(Z,X)',
        'top(Y,Z)',
        'top(Z,Y)',
        'larger(X,Y)',
        'larger(Y,X)',
        'larger(X,Z)',
        'larger(Z,X)',
        'larger(Y,Z)',
        'larger(Z,Y)',
    ]
    head = ["correct(X)"]
    rule = Rule("phi", body=body, head=head, var2domain={"X": "disks", "Y": "disks", "Z": "disks"})

    # train data loading
    X, labels_concepts, labels_tasks, q_names, tower_ids = hanoi_toy_dataset(
        mode='relational', n_samples=n_samples, random_seed=random_seed + fold,
        n_positions=n_positions, n_disks=n_disks, n_sizes=n_sizes)
    train_data = TensorDataset(X, labels_concepts, labels_tasks)
    train_dl = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
    points = Domain("points", [f'{i}' for i in torch.arange(X.shape[1]).tolist()])
    grounder = DomainGrounder({"points": points.constants}, [rule], manifolds_per_rule={"phi": tower_ids})
    indexer = DictBasedIndexer(grounder.ground(), q_names, logic=logic)

    # test data loading
    X_test, labels_concepts_test, labels_task_test, q_names_test, tower_ids_test = hanoi_toy_dataset(
        mode='relational', n_samples=n_samples, random_seed=(random_seed + fold) * 100,
        n_positions=n_positions, n_disks=n_disks, n_sizes=n_sizes)
    points_test = Domain("points", [f'{i}' for i in torch.arange(X_test.shape[1]).tolist()])
    grounder_test = DomainGrounder({"points": points_test.constants}, [rule],
                                   manifolds_per_rule={"phi": tower_ids_test})
    groundings_test = grounder_test.ground()
    indexer_test = DictBasedIndexer(groundings_test, q_names_test, logic=logic)

    model = RelationalCBMDeep(input_features=X.shape[2], n_concepts=len(body),
                              n_classes=len(head), emb_size=emb_size, indexer=indexer,
                              manifold_arity=manifold_arity, learning_rate=learning_rate,
                              concept_names=body, task_names=head, task_weight=task_weight,
                              logic=logic)
    trainer = pl.Trainer(max_epochs=300, enable_progress_bar=False,
                         enable_checkpointing=True,
                         limit_train_batches=limit_batches,
                         limit_val_batches=limit_batches)
    seed_everything(42, workers=True)
    trainer.fit(model=model, train_dataloaders=train_dl)

    # Evaluating on test data
    X, X_test = X.squeeze(0), X_test.squeeze(0)
    labels_concepts, labels_concepts_test = labels_concepts.squeeze(0), labels_concepts_test.squeeze(0)
    labels_tasks, labels_tasks_test = labels_tasks.squeeze(0), labels_task_test.squeeze(0)
    y_true = labels_tasks_test
    model.indexer = indexer_test

    c_preds, y_preds, _, _ = model.forward(X_test)
    concept_accuracy = roc_auc_score(labels_concepts_test, c_preds.detach())
    task_accuracy = roc_auc_score(y_true.squeeze(), y_preds.detach())
    print(f'task accuracy: {task_accuracy:.4f} concept accuracy: {concept_accuracy:.4f}')


if __name__ == '__main__':
    main()

from torch.utils.data import TensorDataset
import numpy as np
from collections import defaultdict
from datasets.gnn_benchmarks import gnn_benchmark_dataset
from model import RelationalCBMDeep
import torch
import os
import pytorch_lightning as pl

from torch_explain.logic.commons import Rule, Domain
from torch_explain.logic.grounding import DomainGrounder
from torch_explain.logic.indexing import DictBasedIndexer
from torch_explain.logic.semantics import ProductTNorm

# disable cuda
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

pl.seed_everything(42)


def main():

    pretrain = True
    learning_rate = 0.01
    batch_size = 1
    limit_batches = 1.0

    dataset_names = ['CORA']

    params = {
        'CORA':{
            'RelationalDCR':  {'emb_size': 32, 'epochs': 500, 'learning_rate': 0.001, "task_weight": 0.5},
            'RelationalCBM': {'emb_size': 32, 'epochs': 3000, 'learning_rate': 0.01, "task_weight": 0.5},
            'RelationalDeepCBM': {'emb_size': 32, 'epochs': 3000, 'learning_rate': 0.01, "task_weight": 0.5},
            'RelationalE2E': {'emb_size': 32, 'epochs': 2000, 'learning_rate': 0.001, "task_weight": 0.5},
            'DeepFeedForward': {'emb_size': 32, 'epochs': 500, 'learning_rate': 0.01, "task_weight": 0.},
            'DeepStochLogCitation': {'emb_size': 32, 'epochs': 500, 'learning_rate': 0.01, "task_weight": 0.5},
        },
    }
    n_seeds = 1
    seeds = [i for i in range(n_seeds)]
    train_size_perc = [1.]

    columns = ["seed", "loss", "concept_accuracy", "task_accuracy"]
    res_tab = defaultdict(lambda: np.zeros([n_seeds, len(columns)]))
    explanations_tab = defaultdict(list)
    for dataset_name in dataset_names:
        for perc_train in train_size_perc:
            for seed in seeds:

                results_root_dir = f"./results/"
                os.makedirs(results_root_dir, exist_ok=True)
                results_dir = f"./results/{dataset_name}/"
                os.makedirs(results_dir, exist_ok=True)
                figures_dir = f"./results/{dataset_name}/figures/"
                os.makedirs(figures_dir, exist_ok=True)

                # RELATIONAL
                X, (train_data, queries_train), (valid_data, queries_valid), (test_data, queries_test), num_classes, manifold_ids = gnn_benchmark_dataset(dataset_name, perc_train=perc_train, pretrain_seed = seed if pretrain else -1)
                train_dl = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
                logic = ProductTNorm()
                documents = Domain("documents", [f'{i}' for i in torch.arange(X.shape[1]).tolist()])
                body = ['class%d(X)' % c for c in range(num_classes)]
                head = ['class%d(Y)' % c for c in range(num_classes)]
                rule = Rule("phi", body=body+head, head=head, var2domain={"X": "documents", "Y": "documents"})
                rule2 = Rule("mutex", body=body, head=[], var2domain={"X": "documents"})
                manifold_arity = len(rule.vars)

                grounder = DomainGrounder({"documents": documents.constants}, [rule, rule2], manifolds_per_rule={"phi": manifold_ids})
                groundings = grounder.ground()
                indexer = DictBasedIndexer(groundings, {"tasks": queries_train, "concepts": queries_train}, logic=logic)

                models = {
                    'RelationalDeepCBM': RelationalCBMDeep(input_features=X.shape[2], n_concepts=len(rule.body),
                                                   n_classes=len(rule.head), emb_size=params[dataset_name]['RelationalDeepCBM']["emb_size"],
                                                   indexer=indexer,
                                                   manifold_arity=manifold_arity, learning_rate=params[dataset_name]['RelationalDeepCBM']["learning_rate"],
                                                   concept_names=body, task_names=head, task_weight=params[dataset_name]['RelationalDeepCBM']["task_weight"],
                                                   logic=logic),
                }

                for model_name, relational_model in models.items():
                    pl.seed_everything(seed)
                    key = (dataset_name, perc_train, model_name)

                    epochs = params[dataset_name][model_name]["epochs"]
                    print("Model: %s" % model_name)
                    print(f'Running epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}')
                    trainer = pl.Trainer(max_epochs=epochs,
                                         limit_train_batches=limit_batches,
                                         limit_val_batches=limit_batches)
                    trainer.fit(model=relational_model, train_dataloaders=train_dl)
                    model_path = os.path.join(results_dir, f'{model_name}.pt')
                    torch.save(relational_model.state_dict(), model_path)

                    groundings = grounder.ground()
                    indexer_test = DictBasedIndexer(groundings, {"tasks": queries_test, "concepts": queries_test}, logic=logic)
                    relational_model.indexer = indexer_test

                    # test model
                    loss, concept_accuracy, task_accuracy = relational_model.validation_step(*test_data, batch_idx=0)
                    c,y,explanations, preds_xformula = relational_model.forward(test_data.tensors[0], explain=True)
                    res_tab[key][seed] = (seed, loss.detach().numpy(), concept_accuracy.detach().numpy(), task_accuracy.detach().numpy())
                    explanations_tab[key] = explanations



    for dataset_name in dataset_names:
        for perc_train in train_size_perc:
            for model_name in models.keys():
                print("Dataset: %s" % dataset_name)
                key = (dataset_name, perc_train, model_name)
                print("Model name: %s" % model_name)
                print("Perc Train Size: %f" % perc_train)
                print("Avg Concept acc:", np.mean(res_tab[key][:, 2]))
                print("Std Concept acc:", np.std(res_tab[key][:, 2]))
                print("Avg acc:", np.mean(res_tab[key][:,3]))
                print("Std acc:", np.std(res_tab[key][:,3]))
                print(explanations_tab[key])

if __name__ == '__main__':
    main()

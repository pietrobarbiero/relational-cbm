from abc import abstractmethod

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss, BCELoss, ModuleList, BCEWithLogitsLoss

from rcbm.logic.indexing import DictBasedIndexer
from rcbm.logic.semantics import ProductTNorm, Logic


def find_best_threshold(y_true, y_pred):
    thresholds = np.arange(0, 1, 0.001)
    to_labels = lambda yp, threshold: (yp >= threshold).numpy().astype('int')
    scores = [f1_score(y_true, to_labels(y_pred, t)) for t in thresholds]
    best_score = np.argmax(scores)
    return thresholds[best_score]


class NeuralNet(pl.LightningModule):
    def __init__(self, input_features: int, n_classes: int, emb_size: int, learning_rate: float = 0.01):
        super().__init__()
        self.input_features = input_features
        self.n_classes = n_classes
        self.emb_size = emb_size
        self.learning_rate = learning_rate
        self.cross_entropy = CrossEntropyLoss(reduction="mean")
        self.bce = BCELoss(reduction="mean")
        self.bce_log = BCEWithLogitsLoss(reduction="mean")

    @abstractmethod
    def forward(self, X):
        raise NotImplementedError

    @abstractmethod
    def _unpack_input(self, I):
        raise NotImplementedError

    def training_step(self, I, batch_idx):
        X, _, y_true = self._unpack_input(I)

        y_preds = self.forward(X)

        loss = self.bce(y_preds.squeeze(), y_true.float().squeeze())
        task_accuracy = accuracy_score(y_true.squeeze(), y_preds > 0.5)
        if self.current_epoch % 10 == 0:
            print(f'{self.__class__} - Train Epoch {self.current_epoch}: task: {task_accuracy:.4f} {loss:.4f}')
        return loss

    def validation_step(self, I, batch_idx):
        X, _, y_true = self._unpack_input(I)
        y_preds = self.forward(X)
        loss = self.bce(y_preds.squeeze(), y_true.float().squeeze())
        self.log("val_acc", roc_auc_score(y_true, y_preds))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


class StandardE2E(NeuralNet):
    def __init__(self, input_features: int, n_classes: int, emb_size: int, learning_rate: float = 0.01):
        super().__init__(input_features, n_classes, emb_size, learning_rate)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_features, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
            torch.nn.Sigmoid()
        )

    def _unpack_input(self, I):
        return I[0], I[1], I[2]

    def forward(self, X, explain=False):
        return self.model(X)


class StandardCBM(StandardE2E):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int,
                 learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 task_weight: float = 0.1):
        super().__init__(input_features, n_classes, emb_size, learning_rate)
        self.n_concepts = n_concepts
        self.concept_names = concept_names
        self.task_names = task_names
        self.task_weight = task_weight
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_features, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )
        self.relation_classifiers = torch.nn.Sequential(torch.nn.Linear(emb_size, n_concepts), torch.nn.Sigmoid())
        self.reasoner = torch.nn.Sequential(
            torch.nn.Linear(n_concepts, n_classes),
            torch.nn.Sigmoid()
        )

    def forward(self, X, explain=False):
        explanation = None
        embeddings = self.encoder(X)
        c_preds = self.relation_classifiers(embeddings)
        y_preds = self.reasoner(c_preds)
        return c_preds, y_preds, explanation

    def training_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        c_preds, y_preds, _ = self.forward(X)

        concept_loss = self.bce(c_preds, c_true.float())
        task_loss = self.bce(y_preds, y_true.float())
        loss = concept_loss + self.task_weight*task_loss

        task_accuracy = roc_auc_score(y_true.squeeze(), y_preds.squeeze().detach())
        concept_accuracy = roc_auc_score(c_true, c_preds.squeeze().detach())
        if self.current_epoch % 10 == 0:
            print(f'{self.__class__} - Train Epoch {self.current_epoch}: task: {task_accuracy:.4f} {task_loss:.4f} '
                  f'concept: {concept_accuracy:.4f} {concept_loss:.4f}')
        return loss

    def validation_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        c_preds, y_preds, _ = self.forward(X)

        concept_loss = self.bce_log(c_preds, c_true.float())
        task_loss = self.bce_log(y_preds, y_true.float())
        loss = concept_loss + self.task_weight*task_loss
        self.log("val_acc", (roc_auc_score(y_true, y_preds) + roc_auc_score(c_true, c_preds)) / 2)
        task_accuracy = roc_auc_score(y_true.squeeze(), y_preds.squeeze().detach())
        concept_accuracy = roc_auc_score(c_true, c_preds.squeeze().detach())
        if self.current_epoch % 10 == 0:
            print(f'{self.__class__} - Valid Epoch {self.current_epoch}: task: {task_accuracy:.4f} {task_loss:.4f} '
                  f'concept: {concept_accuracy:.4f} {concept_loss:.4f}')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


class StandardCBMDeep(StandardCBM):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int,
                 learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 task_weight: float = 0.1):
        super().__init__(input_features, n_concepts, n_classes, emb_size, learning_rate, concept_names, task_names, task_weight)
        self.reasoner = torch.nn.Sequential(
            torch.nn.Linear(n_concepts, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
            torch.nn.Sigmoid()
        )


class StandardDCR(StandardCBM):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int,
                 learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 temperature: float = 10, logic: Logic = ProductTNorm(), explanation_mode: str = 'local',
                 task_weight: float = 0.1):
        super().__init__(input_features, n_concepts, n_classes, emb_size, learning_rate, concept_names, task_names, task_weight)
        self.temperature = temperature
        self.logic = logic
        self.explanation_mode = explanation_mode
        self.reasoner = ConceptReasoningLayer(emb_size, n_concepts=n_concepts, logic=logic,
                                              n_classes=n_classes, temperature=temperature)

    def forward(self, X, explain=True):
        embeddings = self.encoder(X)
        c_preds = self.relation_classifiers(embeddings)
        y_preds = self.reasoner(embeddings, c_preds)
        explanation = None
        if explain:
            explanation = self.reasoner.explain(embeddings, c_preds, self.explanation_mode,
                                                self.concept_names, self.task_names)
        return c_preds, y_preds, explanation


class RelationalE2E(StandardE2E):
    def __init__(self, input_features: int, n_classes: int, emb_size: int, indexer: DictBasedIndexer,
                 manifold_arity: int, learning_rate: float = 0.01,
                 logic: Logic = ProductTNorm()):
        super().__init__(input_features, n_classes, emb_size, learning_rate)
        self.indexer = indexer
        self.logic = logic
        self.manifold_arity = manifold_arity
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_features, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            # torch.nn.LeakyReLU(),
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(emb_size * manifold_arity, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
            torch.nn.Sigmoid()
        )

    def _unpack_input(self, I):
        X, c_true, y_true = I
        c_true = c_true.squeeze(0)
        y_true = y_true.squeeze(0)
        return X, c_true, y_true

    def forward(self, X, explain=False, mode="train"):
        indexer = self.indexer if mode == "train" else self.val_indexer
        X = X.squeeze(0)
        embeddings = self.encoder(X)
        embed_substitutions = indexer.gather_and_concatenate(embeddings, indexer.indexed_subs['phi'], 0)
        grounding_preds = self.predictor(embed_substitutions)
        task_predictions = self.logic.disj_scatter(grounding_preds.view(-1, 1),
                                                   indexer.indexed_heads['phi'],  #TODO: check if it is ok bidimensional heads
                                                   len(indexer.atom_index))
        y_preds = indexer.gather_and_concatenate(task_predictions, indexer.indexed_queries["tasks"], 0)
        return y_preds

    def validation_step(self, I, batch_idx):
        X, _, y_true = self._unpack_input(I)
        y_preds = self.forward(X, mode="val")
        loss = self.bce(y_preds.squeeze(), y_true.float().squeeze())
        self.log("val_acc", roc_auc_score(y_true, y_preds))
        task_accuracy = roc_auc_score(y_true.squeeze(), y_preds.squeeze().detach())
        if self.current_epoch % 10 == 0:
            print(f'{self.__class__} - Valid Epoch {self.current_epoch}: task: {task_accuracy:.4f} {loss:.4f}')
        return loss

class RelationalCBM(RelationalE2E):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int, indexer: DictBasedIndexer,
                 manifold_arity: int, learning_rate: float = 0.01,
                 concept_names: list = None, task_names: list = None, task_weight: float = 0.1, logic=ProductTNorm()):
        super().__init__(input_features, n_classes, emb_size, indexer, manifold_arity, learning_rate, logic)
        self.n_concepts = n_concepts
        self.concept_names = concept_names
        self.task_names = task_names
        self.task_weight = task_weight
        self.classification_thresholds = torch.nn.Parameter(torch.zeros(self.n_classes), requires_grad=False)
        self.relation_classifiers = {}
        for relation_name, relation_arity in indexer.relations_arity.items():
            self.relation_classifiers[relation_name] = torch.nn.Sequential(
                    torch.nn.Linear(emb_size * relation_arity, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, 1),
                    # torch.nn.LeakyReLU(),
                )
        self.reasoner = torch.nn.Sequential(
            torch.nn.Linear(n_concepts, n_classes),
            torch.nn.Sigmoid()
        )

    def _predict_task(self, preds_xformula, embeddings, explain=False, mode="train"):
        indexer = self.indexer if mode == "train" else self.val_indexer
        grounding_preds = self.reasoner(preds_xformula)
        grouped_or = self.logic.disj_scatter(grounding_preds.view(-1, 1),
                                             indexer.indexed_heads['phi'],  #TODO: check if it is ok bidimensional heads
                                             len(indexer.atom_index))
        return grouped_or, None

    def forward(self, X, explain=False, mode="train"):
        indexer = self.indexer if mode == "train" else self.val_indexer

        X = X.squeeze(0)

        # encoding constants
        embeddings = self.encoder(X)
        # embeddings = X

        # relation/concept predictions
        concept_predictions = indexer.predict_relations(encoders=self.relation_classifiers, embeddings=embeddings)
        concept_predictions = torch.sigmoid(concept_predictions)


        preds_xformula = indexer.gather_and_concatenate(params = concept_predictions,
                                                        indices= indexer.indexed_bodies['phi'],
                                                        dim=0)

        # task predictions
        task_predictions, explanations = self._predict_task(preds_xformula, embeddings, explain, mode=mode)
        # task_predictions = torch.sigmoid(task_predictions)

        # lookup
        y_preds = indexer.gather_and_concatenate(task_predictions, indexer.indexed_queries["tasks"], 0)
        c_preds = indexer.gather_and_concatenate(concept_predictions, indexer.indexed_queries["concepts"], 0)


        return c_preds, y_preds, explanations, preds_xformula

    def training_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        c_preds, y_preds, _, _ = self.forward(X)

        concept_loss = self.bce(c_preds.squeeze(), c_true.float())
        task_loss = self.bce(y_preds.squeeze(), y_true.float())
        loss = concept_loss + self.task_weight*task_loss

        if self.current_epoch >= self.trainer.max_epochs - 1:
            for c in range(len(self.task_names)):
                self.classification_thresholds[c] = find_best_threshold(y_true.squeeze(), y_preds.squeeze())

        task_accuracy = roc_auc_score(y_true.squeeze(), y_preds.squeeze().detach())
        concept_accuracy = roc_auc_score(c_true, c_preds.squeeze().detach())
        if self.current_epoch % 10 == 0:
            print(f'{self.__class__} - Train Epoch {self.current_epoch}: task: {task_accuracy:.4f} {task_loss:.4f} '
                  f'concept: {concept_accuracy:.4f} {concept_loss:.4f}')
        return loss

    def validation_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        c_preds, y_preds, _, _ = self.forward(X, mode="val")

        concept_loss = self.bce(c_preds.squeeze(), c_true.float())
        task_loss = self.bce(y_preds.squeeze(), y_true.float())
        loss = concept_loss + self.task_weight*task_loss
        self.log("val_acc", (roc_auc_score(y_true, y_preds) + roc_auc_score(c_true, c_preds)) / 2)
        task_accuracy = roc_auc_score(y_true.squeeze(), y_preds.squeeze().detach())
        concept_accuracy = roc_auc_score(c_true, c_preds.squeeze().detach())
        if self.current_epoch % 10 == 0:
            print(f'{self.__class__} - Valid Epoch {self.current_epoch}: task: {task_accuracy:.4f} {task_loss:.4f} '
                  f'concept: {concept_accuracy:.4f} {concept_loss:.4f}')

        return loss


class PropositionalisedCBM(RelationalCBM):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int, manifold_arity: int,
                 indexer: DictBasedIndexer = None, val_indexer: DictBasedIndexer = None, learning_rate: float = 0.01,
                 concept_names: list = None, task_names: list = None, task_weight: float = 0.1, logic=ProductTNorm()):
        super().__init__(input_features, n_concepts, n_classes, emb_size, indexer, val_indexer, manifold_arity, learning_rate,
                         concept_names, task_names, task_weight, logic)
        self.relation_classifiers = torch.nn.Sequential(
            torch.nn.Linear(emb_size * manifold_arity, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_concepts),
            # torch.nn.LeakyReLU(),
        )
        self.reasoner = torch.nn.Sequential(
            torch.nn.Linear(n_concepts, n_classes),
            torch.nn.Sigmoid()
        )

    def _predict_task(self, preds_xformula, embeddings, explain=False, **kwargs):
        return self.reasoner(preds_xformula), None

    def forward(self, X, explain=False, mode="train"):
        indexer = self.indexer if mode == "train" else self.val_indexer

        X = X.squeeze(0)

        # encoding constants
        embeddings = self.encoder(X)

        # relation/concept predictions
        embedding_constants = indexer.gather_and_concatenate(params = embeddings,
                                                                  indices= indexer.indexed_subs['phi'],
                                                                  dim=0)
        c_preds = self.relation_classifiers(embedding_constants)
        c_preds = torch.sigmoid(c_preds)

        # task predictions
        y_preds, explanations = self._predict_task(c_preds, embeddings, mode=mode)

        return c_preds.ravel(), y_preds.ravel(), explanations, None


class RelationalCBMDeep(RelationalCBM):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int, indexer: DictBasedIndexer,
                 manifold_arity: int, learning_rate: float = 0.01,
                 concept_names: list = None, task_names: list = None, task_weight: float = 0.1, logic=ProductTNorm()):
        super().__init__(input_features, n_concepts, n_classes, emb_size, indexer, manifold_arity, learning_rate,
                         concept_names, task_names, task_weight, logic)
        self.reasoner = torch.nn.Sequential(
            torch.nn.Linear(n_concepts, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
            torch.nn.Sigmoid()
        )


class RelationalDCR(RelationalCBM):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int, indexer: DictBasedIndexer,
                 val_indexer: DictBasedIndexer, manifold_arity: int, learning_rate: float = 0.01,
                 concept_names: list = None, task_names: list = None, temperature: float = 10,
                 logic: Logic = ProductTNorm(), explanation_mode: str = 'local',
                 task_weight: float = 0.1):
        super().__init__(input_features, n_concepts, n_classes, emb_size, indexer, val_indexer, manifold_arity,
                         learning_rate, concept_names, task_names, task_weight, logic)
        self.temperature = temperature
        self.logic = logic
        self.explanation_mode = explanation_mode
        self.reasoner = ConceptReasoningLayer(emb_size * manifold_arity, n_concepts=n_concepts, logic=logic,
                                              n_classes=n_classes, temperature=temperature,
                                              output_sigmoid = True, use_polarity=True)
        # self.automatic_optimization = False

    def _predict_task(self, preds_xformula, embeddings, explain=False, mode="train"):
        indexer = self.indexer if mode == "train" else self.val_indexer

        embed_substitutions = indexer.gather_and_concatenate(embeddings, indexer.indexed_subs['phi'], 0)
        grounding_preds = self.reasoner(embed_substitutions, preds_xformula)
        task_predictions = self.logic.disj_scatter(grounding_preds.view(-1, 1),
                                             indexer.indexed_heads['phi'],
                                             len(indexer.atom_index))
        explanations = None
        if explain:
            explanations = self.reasoner.explain(embed_substitutions, preds_xformula,
                                                 mode=self.explanation_mode, concept_names=self.concept_names,
                                                 class_names=self.task_names,
                                                 classification_threshold=self.classification_thresholds[0])
        return task_predictions, explanations


    # def training_step(self, I, batch_idx):
    #     self.train()
    #     concept_opt, task_opt = self.optimizers()
    #
    #
    #     X, c_true, y_true = self._unpack_input(I)
    #
    #     c_preds, y_preds, _, _ = self.forward(X)
    #
    #
    #     c_preds, y_preds, _, _ = self.forward(X)
    #     concept_loss = self.bce(c_preds.squeeze(), c_true.float())
    #     concept_opt.zero_grad()
    #     self.manual_backward(concept_loss)
    #     concept_opt.step()
    #
    #     # task_loss_dcr = 0.
    #     c_preds, y_preds, _, _ = self.forward(X)
    #     task_loss = self.bce(y_preds.squeeze(), y_true.float())
    #     task_opt.zero_grad()
    #     self.manual_backward(self.task_weight*task_loss)
    #     task_opt.step()
    #
    #     loss = concept_loss + self.task_weight*task_loss
    #
    #
    #     concept_accuracy = accuracy_score(c_true.squeeze(), c_preds.squeeze() > 0.5)
    #     task_accuracy_dcr = accuracy_score(y_true.squeeze(), y_preds.squeeze() > 0.5)
    #     print(f'Epoch {self.current_epoch}: loss: {loss:.4f} concept_accuracy: {concept_accuracy:.4f} task accuracy: {task_accuracy_dcr:.4f}')
    #     return loss
    #
    # def validation_step(self, I, batch_idx):
    #     self.eval()
    #     X, c_true, y_true = self._unpack_input(I)
    #
    #     c_preds, y_preds, _, _ = self.forward(X)
    #
    #     concept_loss = self.bce(c_preds.squeeze(), c_true.float())
    #     task_loss = self.bce(y_preds.squeeze(), y_true.float())
    #     loss = concept_loss + 0.5*task_loss
    #     return loss
    #
    # def configure_optimizers(self):
    #     concept_opt = torch.optim.AdamW(self.parameters(), lr=1e-3)
    #     dcr_opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    #     return concept_opt, dcr_opt
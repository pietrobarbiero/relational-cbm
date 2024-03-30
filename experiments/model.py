from abc import abstractmethod

import torch
from sklearn.metrics import accuracy_score
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss, BCELoss,NLLLoss, ModuleList, BCEWithLogitsLoss

from rcbm.logic.indexing import DictBasedIndexer
from rcbm.logic.semantics import ProductTNorm, Logic
# from torch_explain.nn.concepts import ConceptReasoningLayer


def categorical_cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()

class NeuralNet(pl.LightningModule):
    def __init__(self, input_features: int, n_classes: int, emb_size: int, learning_rate: float = 0.01):
        super().__init__()
        self.input_features = input_features
        self.n_classes = n_classes
        self.emb_size = emb_size
        self.learning_rate = learning_rate
        self.cross_entropy = NLLLoss(reduction="mean")
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
        print(f'Epoch {self.current_epoch}: task accuracy: {task_accuracy:.4f}')
        return loss

    # def validation_step(self, I, batch_idx):
    #     X, _, y_true = self._unpack_input(I)
    #     y_preds = self.forward(X)
    #     loss = self.bce(y_preds.squeeze(), y_true.float().squeeze())
    #     return loss

    def validation_step(self, I, batch_idx):
        X, _, y_true = self._unpack_input(I)

        y_preds = self.forward(X)

        task_loss = self.bce(y_preds.squeeze(), y_true.float())

        y_true =  torch.argmax(y_true.view(-1, self.n_classes), dim=-1)
        y_preds = torch.argmax( y_preds.view(-1, self.n_classes), dim=-1)

        task_accuracy = torch.mean((y_true == y_preds).float())
        print(f'Epoch {self.current_epoch}: task eval accuracy: {task_accuracy:.4f}')
        return task_loss, torch.tensor(0.), task_accuracy


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
                 learning_rate: float = 0.01, concept_names: list = None, task_names: list = None):
        super().__init__(input_features, n_classes, emb_size, learning_rate)
        self.n_concepts = n_concepts
        self.concept_names = concept_names
        self.task_names = task_names
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_features, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )
        self.relation_classifiers = torch.nn.Sequential(torch.nn.Linear(emb_size, n_concepts))
        self.reasoner = torch.nn.Sequential(
            torch.nn.Linear(n_concepts, n_classes),
            # torch.nn.Sigmoid()
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

        concept_loss = self.bce_log(c_preds, c_true.float())
        task_loss = self.bce_log(y_preds, y_true.float())
        loss = concept_loss + 0.5*task_loss

        task_accuracy = accuracy_score(y_true.squeeze(), y_preds > 0.)
        concept_accuracy = accuracy_score(c_true, c_preds > 0.)
        print(f'Epoch {self.current_epoch}: task accuracy: {task_accuracy:.4f} concept accuracy: {concept_accuracy:.4f}')
        return loss

    def validation_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        c_preds, y_preds, _ = self.forward(X)

        concept_loss = self.bce_log(c_preds, c_true.float())
        task_loss = self.bce_log(y_preds, y_true.float())
        loss = concept_loss + 0.5*task_loss
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
    def __init__(self, input_features: int, n_classes: int, emb_size: int, indexer: DictBasedIndexer, manifold_arity: int,
                 learning_rate: float = 0.01, logic: Logic = ProductTNorm()):
        super().__init__(input_features, n_classes, emb_size, learning_rate)
        self.indexer = indexer
        self.logic = logic
        self.manifold_arity = manifold_arity
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_features, emb_size),
            torch.nn.LeakyReLU(),
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(emb_size * manifold_arity, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
            torch.nn.Sigmoid()
        )

    def _unpack_input(self, I):
        X, c_true, y_true = I
        c_true = c_true.squeeze(0)
        y_true = y_true.squeeze(0)
        return X, c_true, y_true

    def forward(self, X, explain=False):
        X = X.squeeze(0)
        embeddings = self.encoder(X)
        embed_substitutions = self.indexer.gather_and_concatenate(embeddings, self.indexer.indexed_subs['phi'], 0)
        grounding_preds = self.predictor(embed_substitutions)
        task_predictions = self.logic.disj_scatter(grounding_preds.view(-1, 1),
                                                   self.indexer.indexed_heads['phi'],  #TODO: check if it is ok bidimensional heads
                                                   len(self.indexer.atom_index))
        y_preds = self.indexer.gather_and_concatenate(task_predictions, self.indexer.indexed_queries["tasks"], 0)
        if explain:
            return None, y_preds, None, None
        return y_preds






class RelationalCBM(RelationalE2E):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int, indexer: DictBasedIndexer,
                 manifold_arity: int, learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 task_weight: float = 0.1, logic=ProductTNorm()):
        super().__init__(input_features, n_classes, emb_size, indexer, manifold_arity, learning_rate, logic)
        self.n_concepts = n_concepts
        self.concept_names = concept_names
        self.task_names = task_names
        self.task_weight = task_weight
        self.classification_thresholds = []
        self.relation_classifiers = {}
        for relation_name, relation_arity in indexer.relations_arity.items():
            self.relation_classifiers[relation_name] = torch.nn.Sequential(
                    torch.nn.Linear(emb_size, 1),
                )
        self.reasoner = torch.nn.Sequential(
            torch.nn.Linear(n_concepts, n_classes),
            torch.nn.Sigmoid()
        )

    def _predict_task(self, preds_xformula, embeddings, explain=False):

        grounding_preds = self.reasoner(preds_xformula)
        task_predictions = self.logic.disj_scatter(grounding_preds.view(-1, 1),
                                                   self.indexer.indexed_heads['phi'].view(-1, 1),
                                                   len(self.indexer.atom_index))
        return task_predictions, None

    def forward(self, X, explain=False):
        X = X.squeeze(0)

        # encoding constants
        embeddings = self.encoder(X)
        # embeddings = X

        # relation/concept predictions
        concept_logits = self.indexer.predict_relations(encoders = self.relation_classifiers,embeddings = embeddings)
        concepts_logits_mutually_exclusive_groups = concept_logits[self.indexer.indexed_bodies['mutex']].squeeze(-1)
        concepts_predictions_mutually_exclusive_groups = torch.softmax(concepts_logits_mutually_exclusive_groups, dim=-1)

        concept_predictions = torch.sigmoid(concept_logits)
        concept_predictions = torch.scatter(input=concept_predictions,
                                            index=self.indexer.indexed_bodies["mutex"].view(-1, 1),
                                            src=concepts_predictions_mutually_exclusive_groups.view(-1, 1),
                                            dim=0)

        task_predictions = concept_predictions
        for i in range(1):
            preds_xformula = self.indexer.gather_and_concatenate(params = task_predictions,
                                                                 indices= self.indexer.indexed_bodies['phi'],
                                                                 dim=0)

            # task predictions
            task_predictions, explanations = self._predict_task(preds_xformula, embeddings, explain=explain)
            # task_predictions = torch.maximum(task_predictions, concept_predictions)
            # task_predictions = torch.sigmoid(task_predictions)

        # lookup
        y_preds = self.indexer.gather_and_concatenate(task_predictions, self.indexer.indexed_queries["tasks"], 0)
        c_preds = self.indexer.gather_and_concatenate(concept_predictions, self.indexer.indexed_queries["concepts"], 0)


        return c_preds, y_preds, explanations, preds_xformula

    def training_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        c_preds, y_preds, _, _ = self.forward(X)

        concept_loss = categorical_cross_entropy(c_preds.view(-1, self.n_classes), c_true.view(-1, self.n_classes))
        task_loss = self.bce(y_preds.squeeze(), y_true.float())
        loss = concept_loss + self.task_weight*task_loss

        c_true =  torch.argmax(c_true.view(-1, self.n_classes), dim=-1)
        y_true =  torch.argmax(y_true.view(-1, self.n_classes), dim=-1)
        c_preds = torch.argmax( c_preds.view(-1, self.n_classes), dim=-1)
        y_preds = torch.argmax( y_preds.view(-1, self.n_classes), dim=-1)

        task_accuracy = torch.mean((y_true == y_preds).float())
        concept_accuracy = torch.mean((c_true == c_preds).float())
        print(f'Epoch {self.current_epoch}: task accuracy: {task_accuracy:.4f} concept accuracy: {concept_accuracy:.4f}')
        return loss

    def validation_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        c_preds, y_preds, _, _ = self.forward(X)

        concept_loss = categorical_cross_entropy(c_preds.view(-1, self.n_classes), c_true.view(-1, self.n_classes))
        task_loss = self.bce(y_preds.squeeze(), y_true.float())
        loss = concept_loss + self.task_weight*task_loss

        c_true =  torch.argmax(c_true.view(-1, self.n_classes), dim=-1)
        y_true =  torch.argmax(y_true.view(-1, self.n_classes), dim=-1)
        c_preds = torch.argmax( c_preds.view(-1, self.n_classes), dim=-1)
        y_preds = torch.argmax( y_preds.view(-1, self.n_classes), dim=-1)

        task_accuracy = torch.mean((y_true == y_preds).float())
        concept_accuracy = torch.mean((c_true == c_preds).float())
        print(f'Epoch {self.current_epoch}: task eval accuracy: {task_accuracy:.4f} concept eval accuracy: {concept_accuracy:.4f}')
        return loss, concept_accuracy, task_accuracy


class RelationalCBMDeep(RelationalCBM):

    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int, indexer: DictBasedIndexer,
                 manifold_arity: int, learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 task_weight: float = 0.1, logic=ProductTNorm()):
        super().__init__(input_features, n_concepts, n_classes, emb_size, indexer, manifold_arity, learning_rate, concept_names, task_names, task_weight, logic)
        self.reasoner = torch.nn.Sequential(
            torch.nn.Linear(n_concepts, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
            torch.nn.Sigmoid()
        )


class RelationalDCR(RelationalCBM):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int, indexer: DictBasedIndexer,
                 manifold_arity: int, learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 task_weight:float = 1., temperature: float = 10, logic: Logic = ProductTNorm(), explanation_mode: str = 'global'):
        super().__init__(input_features, n_concepts, n_classes, emb_size, indexer, manifold_arity, learning_rate,
                         concept_names, task_names, task_weight, logic)
        self.temperature = temperature
        self.logic = logic
        self.explanation_mode = explanation_mode
        self.reasoner = ConceptReasoningLayer(emb_size * manifold_arity, n_concepts=n_concepts, logic=logic,
                                              n_classes=n_classes, temperature=temperature,
                                              output_sigmoid = True, use_polarity=True)

    def _predict_task(self, preds_xformula, embeddings, explain=False):

        embed_substitutions = self.indexer.gather_and_concatenate(embeddings, self.indexer.indexed_subs['phi'], 0)
        grounding_preds = self.reasoner(embed_substitutions, preds_xformula)
        task_predictions = self.logic.disj_scatter(grounding_preds.view(-1,1),
                                             self.indexer.indexed_heads['phi'].view(-1,1),
                                             len(self.indexer.atom_index))
        explanations = None
        if explain:
            explanations = self.reasoner.explain(embed_substitutions, preds_xformula,
                                                 self.explanation_mode, self.concept_names, self.task_names)
        return task_predictions, explanations


class SumProductCitation(torch.nn.Module):

    def __init__(self,n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.rule_tensor_indices = torch.tensor([[[i]] for i in range(n_classes)])

    def forward(self, i):
        # i = i.detach()
        x = i[:, self.rule_tensor_indices]
        y = torch.prod(x, dim=-1)
        y = torch.sum(y,dim=-1, keepdim=True)
        return y


class DeepStochLogCitation(RelationalCBM):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int, indexer: DictBasedIndexer,
                 manifold_arity: int, learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 task_weight: float = 0.1, logic=ProductTNorm()):
        super().__init__(input_features, n_concepts, n_classes, emb_size, indexer, manifold_arity, learning_rate,
                         concept_names, task_names, task_weight, logic)
        self.reasoner = SumProductCitation(n_classes)

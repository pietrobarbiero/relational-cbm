import copy
from collections import defaultdict, OrderedDict
from typing import List, Dict, Tuple, Set
import torch
from torch import Tensor
from rcbm.logic.commons import Rule, FOL, Predicate, Domain
from rcbm.logic.semantics import Logic





class AbstractIndexer():

    def __init__(self, fol: FOL, groundings: Dict[str, List[Tuple[Tuple, Tuple]]],
                 queries: Dict[str, List[str]], logic: Logic):
        pass

    def lookup_query(self, source: torch.Tensor, key: str) -> torch.Tensor:
        pass


    def apply_index_formulas_constants(self, constant_embedddings, formula_id):
        pass

    def apply_index_formulas_relations(self, atom_predictions, formula_id):
        pass

    def group_or(self, grounding_predictions, formula_id):
        pass

    def encode_atoms(self, X, relation_encoders):
        pass


class DictBasedIndexer():

    def __init__(self, groundings: Dict[str, List[Tuple[Tuple, Tuple]]],
                 queries: Dict[str, List[str]],
                 logic: Logic):


        queries = {k: [query_str_to_tuple(atom) for atom in v] for k,v in queries.items()}
        self.queries = queries
        self.groundings = groundings


        # Store all atoms and all groundinngs
        self.atoms_per_relation: Dict[str, Set[Tuple]] = self.group_all_atoms_by_relation(queries, groundings)

        # Sorting and indexing relations
        self.relation_index: Dict[str, int] = self.index_relations(self.atoms_per_relation)

        # Sorting and indexing atoms following relations order.
        #(OrderedDict[Tuple, int], OrderedDict[str, List]):
        self.atom_index, self.indices_tuples, self.relations_arity = self.index_atoms(self.atoms_per_relation, self.relation_index)

        #Index Queries
        self.indexed_queries = self.index_queries(self.queries, self.atom_index)

        #Index Groundings
        self.indexed_heads, self.indexed_bodies, self.indexed_subs = self.index_groundings(self.groundings, self.atom_index)





    def predict_relations(self, encoders, embeddings):
        predictions = []
        for rel_name, _ in self.relation_index.items():
            relation_classifier = encoders[rel_name]
            embedding_constants = self.gather_and_concatenate(params = embeddings,
                                                              indices = self.indices_tuples[rel_name],
                                                              dim=0)
            predictions.append(relation_classifier(embedding_constants))
        predictions = torch.cat(predictions, dim=0)
        return predictions

    def gather_and_concatenate(self, params, indices, dim):
        del dim
        return params[indices].view(indices.shape[0], -1 )


    def index_queries(self, queries, atom_index):
        indexed_queries = {}
        for category, tpls in queries.items():
            indexed_queries[category] = torch.tensor([atom_index[tpl] for tpl in tpls])
        return indexed_queries

    def index_groundings(self, groundings, atom_index):
        indexed_heads = {}
        indexed_bodies = {}
        indexed_subs = {}

        for id_formula, (grds, subs) in groundings.items():
            indexed_head = []
            indexed_body = []
            for head, body in grds:
                indexed_head.append([atom_index[at] for at in head])
                indexed_body.append([atom_index[at] for at in body])

            indexed_sub = []
            for s in subs:
                indexed_sub.append([int(d) for d in s])

            indexed_heads[id_formula] = torch.tensor(indexed_head)
            indexed_bodies[id_formula] = torch.tensor(indexed_body)
            indexed_subs[id_formula] = torch.tensor(indexed_sub)
        return indexed_heads, indexed_bodies, indexed_subs

    def index_relations(self, atoms_per_relation):
        sorted_relations = sorted(atoms_per_relation.keys())
        index_relations = OrderedDict()

        for rel in sorted_relations:
            index_relations[rel] = len(index_relations)
        return index_relations


    def index_atoms(self, atoms_per_relation, relation_index) -> (Dict[Tuple, int], Dict[str, List]):

        index_atoms = OrderedDict()
        index_tuples = OrderedDict()
        relations_arity = {}


        for rel, id in relation_index.items():
            all_atoms_of_rel = sorted(atoms_per_relation[rel])
            tuples_of_rel = []
            relations_arity[rel] = len(all_atoms_of_rel[0]) - 1
            for atom in all_atoms_of_rel:
                index_atoms[atom] = len(index_atoms)
                tuples_of_rel.append([int(k) for k in atom[1:]])
            index_tuples[rel] = torch.tensor(tuples_of_rel)
        return index_atoms, index_tuples, relations_arity



    def add_atoms_to_dict(self, atoms, atoms_per_relation):
        for atom in atoms:
            r = atom[0]
            atoms_per_relation[r].add(atom)
        return atoms_per_relation


    def group_all_atoms_by_relation(self, queries, groundings):

        atoms_by_relation = defaultdict(set)

        for category, queries_per_category in queries.items():
            atoms_by_relation = self.add_atoms_to_dict(queries_per_category, atoms_by_relation)

        for id_formula, (grds, subs) in groundings.items():
            for head, body in grds:
                atoms_by_relation = self.add_atoms_to_dict(head, atoms_by_relation)
                atoms_by_relation = self.add_atoms_to_dict(body, atoms_by_relation)

        return atoms_by_relation






def tuple_to_indexes(t, atom_index, query_index, relation_index, relation_arity):
    # TODO: add documentation
    if t in query_index:
        atom_index[t] = query_index[t]
    else:
        query_index[t] = len(query_index)
        atom_index[t] = query_index[t]
    if t[0] not in relation_index:
        relation_index[t[0]] = len(relation_index)
        relation_arity[t[0]] = len(t) - 1
    return [(atom_index[t], relation_index[t[0]], int(i), j) for j, i in enumerate(t[1:])]


def query_str_to_tuple(query_str: str) -> Tuple:
    query_name = query_str.split('(')[0]
    query_input_ids = query_str.split('(')[1][:-1]
    # query_tuple = tuple([query_name] + [int(i) for i in query_input_ids.split(',')])
    query_tuple = tuple([query_name] + [i for i in query_input_ids.split(',')])
    return query_tuple


# def tuple_str_to_int(t: Tuple) -> Tuple:
#     return tuple(t[0]) + tuple([int(i) for i in t[1:]])


def sort_index(index: torch.Tensor, pos: List[int]) -> torch.Tensor:
    # TODO: check whether we need extra arguments to control the columns to sort
    # We need them to be sorted first, by relation (column=1) and, then, by position (column=3)
    for p in pos:
        index = index[torch.argsort(index[:, p], stable=True)]
    # index = index[torch.argsort(index[:, 3])]
    # index = index[torch.argsort(index[:, 1], stable=True)]
    return index

    # _, indices = torch.sort(t[:, dim])
    # t = t[indices]
    # ids = t[:, dim].unique()
    # mask = t[:, None, dim] == ids
    # splits = torch.argmax(mask.float(), dim=0)
    # r = torch.tensor_split(t, splits[1:])
    # return r

def group_by_no_for(groupby_values, tensor_to_group=None, dim=None):
    # TODO: add documentation
    if tensor_to_group is None:
        tensor_to_group = groupby_values
    if dim is not None:
        _, sorted_groupby_indices = torch.sort(groupby_values[:, dim], stable=True)
    else:
        _, sorted_groupby_indices = torch.sort(groupby_values, stable=True)
    sorted_groupby_values = groupby_values[sorted_groupby_indices]

    if dim is not None:
        split_group = sorted_groupby_values
        unique_groupby_values = sorted_groupby_values[:, dim].unique()
        mask_for_split = sorted_groupby_values[:, None, dim] == unique_groupby_values
    else:
        split_group = tensor_to_group[sorted_groupby_indices]
        unique_groupby_values = groupby_values.unique()
        mask_for_split = sorted_groupby_values[:, None] == unique_groupby_values

    splits = torch.argmax(mask_for_split.float(), dim=0)
    return torch.tensor_split(split_group, splits[1:])


def intersect_1d_no_loop(a: torch.tensor, b: torch.tensor):
    return a[(a.view(1, -1) == b.view(-1, 1)).any(dim=0)]


class Indexer:

    def __init__(self, groundings: Dict[str, List[Tuple[Tuple, Tuple]]],
                 queries: Dict[str, List[str]], logic: Logic):
        # TODO: add documentation with simple example of how to use the class

        self.groundings = groundings
        self.queries = queries
        # dictionary of unique grounded queries: {query_tuple: index}
        # a query tuple is a tuple of the form: ('relation_name', 'input_id_1', ..., 'input_id_n')
        self.unique_query_index = {}

        # dictionary containing the index of all the unique grounded atoms (query and not!): {relation_tuple: index}
        # a relation tuple is a tuple of the form: ('relation_name', 'input_id_1', ..., 'input_id_n')
        # the dictionary is initialized with the unique grounded queries
        # the dictionary is updated with the other grounded atoms found in grounded rules
        self.grounded_relation_index = {}

        # dictionary containing the index of unique grounded atoms found in the rules: {relation_tuple: index}
        # a relation tuple is a tuple of the form: ('relation_name', 'input_id_1', ..., 'input_id_n')
        # the dictionary may not contain all the queries (it contains only queries found in the rules)
        self.unique_atom_index = {}

        # dictionary containing the index of all relation names: {'relation_name': index}
        # the dictionary is updated while looping over the grounded rules
        self.relation_index = {}
        self.relation_arity = {}

        # dictionary containing the index of all the unique bodies found in grounded rules: {body_tuple: index}
        # a body tuple is a tuple of the form: (('relation_name', 'input_id_1', ..., 'input_id_n'), ...)
        # the dictionary is updated while looping over the grounded rules
        self.bodies_index = {}

        # dictionary containing the index of all the unique grounded rules: {'formula_name': index}
        # the dictionary is updated while looping over the grounded rules
        self.formulas_index = {}

        self.logic = logic
        self.indices = {}
        self.indices_groups = {}
        self.supervised_queries_ids = None
        self.indices_head = set()

    def index_all(self) -> Tuple[Dict[str, Tensor], Dict[str, List[Tensor]]]:
        init_index_queries = self.init_index_queries()
        index_atoms = sort_index(torch.tensor(self.index_atoms()), pos=[3, 1])
        index_formulas, dict_formula_tuples = self.index_formulas()
        index_formulas = sort_index(torch.tensor(index_formulas), pos=[3, 1])
        self.indices = {
            'queries': init_index_queries,  # [query_id_1, ..., query_id_n]
            'atoms': index_atoms,  # [(atom_index, relation_index, input_id, position), ...]
            'formulas': index_formulas,# [(body_index, formula_index, grounded_relation_index, position, head_index), ...]
            "substitutions": {k:torch.tensor(v) for k,v in dict_formula_tuples.items()} #dict[formula_id, List[Tuple[costant_id]]]
        }
        self.indices_groups = {
            'atoms': group_by_no_for(self.indices['atoms'], dim=1),
            'formulas': group_by_no_for(self.indices['formulas'], dim=1),
        }
        self.supervised_queries_ids = torch.tensor(list(self.unique_query_index.values()))
        return self.indices, self.indices_groups

    def lookup_query(self, source: torch.Tensor, key: str) -> torch.Tensor:
        ids = self.indices['queries'][key]
        return source[ids]

    def get_supervised_slice(self, y, y_ids):
        supervised_y_ids = intersect_1d_no_loop(y_ids, self.supervised_queries_ids)
        return y[supervised_y_ids]

    def apply_index(self, X, index_name, group_id):
        # TODO: add documentation
        # rel_id = index[0, 1]
        tuples = group_by_no_for(self.indices_groups[index_name][group_id], dim=0)
        tuples = torch.stack(tuples, dim=0)
        if tuples.shape[-1] > 4:
            atom_ids = tuples[:, 0, -1]
        else:
            atom_ids = tuples[:, 0, 0]
        tuples = tuples[:, :, 2]

        return X[tuples].view(tuples.shape[0], -1), tuples, atom_ids

    def apply_index_atoms(self, X, group_id):
        # TODO: add documentation
        # rel_id = index[0, 1]
        tuples = group_by_no_for(self.indices_groups["atoms"][group_id], dim=0)
        tuples = torch.stack(tuples, dim=0)
        atom_ids = tuples[:, 0, 0]
        tuples = tuples[:, :, 2]

        return X[tuples].view(tuples.shape[0], -1), tuples, atom_ids

    def apply_index_formulas_constants(self, constant_embedddings, formula_id):
        group_id = self.formulas_index[formula_id]
        substitutions = self.indices["substitutions"][group_id]
        return constant_embedddings[substitutions].view(substitutions.shape[0], -1)

    def apply_index_formulas_relations(self, atom_predictions, formula_id):
        group_id = self.formulas_index[formula_id]
        tuples = group_by_no_for(self.indices_groups["formulas"][group_id], dim=0)
        tuples = torch.stack(tuples, dim=0)
        tuples = tuples[:, :, 2]
        return atom_predictions[tuples].view(tuples.shape[0], -1)

    def group_or(self, grounding_predictions, formula_id):

        group_id = self.formulas_index[formula_id]
        tuples = group_by_no_for(self.indices_groups["formulas"][group_id], dim=0)
        tuples = torch.stack(tuples, dim=0)
        head_ids = tuples[:, 0, -1]

        grouped_or = self.logic.disj_scatter(grounding_predictions, head_ids, len(self.unique_atom_index))
        # grouped_max = self.logic.max_scatter(grounding_predictions, head_ids, len(self.unique_atom_index))
        #
        # max_groundings = torch.where(grouped_max[head_ids] == grounding_predictions)[0]

        return grouped_or#, max_groundings

        # grounding_predictions_agg = torch.zeros(len(self.unique_atom_index), 1) # TODO: what is the neutral element of the T-norm?
        # return grounding_predictions_agg.scatter_reduce(0, head_ids.view(-1, 1), grounding_predictions, reduce='amax') # TODO: do this operation according to T-norm
        # grounding_predictions = torch.log(grounding_predictions)
        # grounding_predictions_agg = grounding_predictions_agg.scatter_reduce(0, head_ids.view(-1, 1), grounding_predictions, reduce='sum')
        # return torch.sigmoid(grounding_predictions_agg)

    def get_all_rules_query(self, X, ctrue_xformula, y_true, preds_xformula, y_preds, explanations):
        explanations_grouped_per_query = {}
        for grounding_id in range(len(self.groundings['phi'][0])):
            head, body = self.groundings['phi'][0][grounding_id]
            query = head[0][0] + f'({head[0][1]})'
            qid = self.unique_query_index[head[0]]
            if query not in explanations_grouped_per_query:
                if explanations[grounding_id]['explanation'] != '' and \
                        ((y_preds[qid] > 0.5) == y_true[qid]) and \
                        ((preds_xformula[grounding_id] > 0.5) == ctrue_xformula[grounding_id]).all():
                    c1, c2 = self.groundings['phi'][1][grounding_id]
                    yidx = self.queries['tasks'].index(query)
                    explanations_grouped_per_query[query] = {}
                    explanations_grouped_per_query[query]['source'] = [X[int(c1)]]
                    explanations_grouped_per_query[query]['dest'] = [X[int(c2)]]
                    explanations_grouped_per_query[query]['pred'] = [y_preds[yidx] > 0.5]
                    explanations_grouped_per_query[query]['cpred'] = [preds_xformula[grounding_id]]
                    explanations_grouped_per_query[query]['rule'] = [(body, explanations[grounding_id])]
            else:
                if explanations[grounding_id]['explanation'] != '' and \
                        ((y_preds[qid] > 0.5) == y_true[qid]) and \
                        ((preds_xformula[grounding_id] > 0.5) == ctrue_xformula[grounding_id]).all():
                    c1, c2 = self.groundings['phi'][1][grounding_id]
                    yidx = self.queries['tasks'].index(query)
                    explanations_grouped_per_query[query]['source'].append(X[int(c1)])
                    explanations_grouped_per_query[query]['dest'].append(X[int(c2)])
                    explanations_grouped_per_query[query]['pred'].append(y_preds[yidx] > 0.5)
                    explanations_grouped_per_query[query]['cpred'].append(preds_xformula[grounding_id])
                    explanations_grouped_per_query[query]['rule'].append((body, explanations[grounding_id]))

        return explanations_grouped_per_query

    def get_max_rules_query(self, X, ctrue_xformula, y_true, preds_xformula, y_preds, explanations):
        explanations_grouped_per_query = defaultdict(lambda: {'max': 0})
        for grounding_id in range(len(self.groundings['phi'][0])):
            head, body = self.groundings['phi'][0][grounding_id]
            query = head[0][0] + f'({head[0][1]})'
            yidx = self.queries['tasks'].index(query)
            if explanations[grounding_id]['explanation'] != '' and \
                    ((y_preds[yidx] > 0.5) == y_true[yidx]) and \
                    ((preds_xformula[grounding_id] > 0.5) == ctrue_xformula[grounding_id]).all() and \
                    (y_preds[yidx] > explanations_grouped_per_query[query]['max']):

                c1, c2 = self.groundings['phi'][1][grounding_id]
                explanations_grouped_per_query[query]['source'] = [X[int(c1)]]
                explanations_grouped_per_query[query]['dest'] = [X[int(c2)]]
                explanations_grouped_per_query[query]['pred'] = [y_preds[yidx] > 0.5]
                explanations_grouped_per_query[query]['cpred'] = [preds_xformula[grounding_id]]
                explanations_grouped_per_query[query]['rule'] = [(body, explanations[grounding_id])]
                explanations_grouped_per_query[query]['max'] = y_preds[yidx]

        return explanations_grouped_per_query

    def init_index_queries(self) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        Initialize the index of unique grounded queries (a.k.a. the concepts/relations to be supervised).
        This function loops over the input queries of the form: ['relation_name(input_id_1, ..., input_id_n)', ...]
        and creates a dictionary of unique grounded queries: {query_tuple: index}
        where query_tuple is a tuple of the form: ('relation_name', 'input_id_1', ..., 'input_id_n').
        :return: index of unique grounded queries
        """
        indices_queries_dict = {}
        for k in ['tasks', 'concepts']: # FIXME: this is really dangerous, it may create indixing problems
            v = self.queries[k]
            indices_queries = []
            for q in v:
                query_tuple = query_str_to_tuple(q)

                # append query to unique_tuples if not already present
                if query_tuple not in self.unique_query_index:
                    self.unique_query_index[query_tuple] = len(self.unique_query_index)

                # append query ID to indices_queries
                indices_queries.append(self.unique_query_index[query_tuple])

            indices_queries_dict[k] = indices_queries
        return indices_queries_dict

    def index_atoms(self) -> List[Tuple[int, int, int, int]]:
        """
        Index all the grounded atoms found in the bodies/heads of grounded rules.

        This function loops over the grounded rules of the form: {'formula_name': [((head_tuple_1, ...), (body_tuple_1, ...)), ...], ...}.
        where each tuple is of the form: ('relation_name', 'input_id_1', ..., 'input_id_n').

        This function creates an index of all the grounded atoms of the form: (atom_index, relation_index, input_id, position).
        The atom_index is given by self.unique_atom_index.
        The relation_index is given by self.relation_index.
        The input_id is given by int(input_id) in the tuple.
        The position is given by the position of the input_id in the tuple.
        :return: index of unique grounded atoms
        """
        # initialize the index of unique grounded atoms with the unique grounded queries
        # this way the grounded_relation_index indeces are aligned with the unique_query_index indeces
        # FIXME: do this with self.unique_atom_index!
        # the grounded relation index should only be used for the grounded relations
        self.grounded_relation_index = copy.deepcopy(self.unique_query_index)

        # loop over all atoms found in heads/bodies of grounded rules
        indices_atoms = []
        for k, (groundings, _) in self.groundings.items():
            for head, body in groundings:

                for h_tuple in head:
                    # skip if the current atom was already indexed
                    # otherwise generate index of the form: (atom_index, relation_index, input_id, position)
                    # and append index to indices_atoms
                    if h_tuple not in self.unique_atom_index:
                        # TODO: check whether we actually need all these indexes
                        idx = tuple_to_indexes(h_tuple, self.unique_atom_index, self.grounded_relation_index, self.relation_index, self.relation_arity)
                        indices_atoms.extend(idx)

                    self.indices_head.add(h_tuple)

                for b_tuple in body:
                    # skip if the current atom was already indexed
                    # otherwise generate index of the form: (atom_index, relation_index, input_id, position)
                    # and append index to indices_atoms
                    if b_tuple not in self.unique_atom_index:
                        idx = tuple_to_indexes(b_tuple, self.unique_atom_index, self.grounded_relation_index, self.relation_index, self.relation_arity)
                        indices_atoms.extend(idx)

        return indices_atoms

    def index_formulas(self) -> List[Tuple[int, int, int, int, int]]:
        """
        Index all the grounded formulas.

        This function loops over the grounded rules of the form: {'formula_name': [((head_tuple_1, ...), (body_tuple_1, ...)), ...], ...}.
        where each tuple is of the form: ('relation_name', 'input_id_1', ..., 'input_id_n').

        This function creates an index of all the grounded formulas of the form: (body_index, formula_index, grounded_relation_index, position, head_index).
        The body_index is given by self.bodies_index.
        The formula_index is given by self.formulas_index.
        The grounded_relation_index is given by self.grounded_relation_index.
        The position is given by the position of the body_tuple in the formula.
        The head_index is given by self.unique_query_index.

        :return: index of unique grounded formulas
        """
        # loop over all formulas
        indices_formulas = []
        indices_tuples_formulas = {}
        indices_formulas_dict = {}
        for k, (groundings, substitutions) in self.groundings.items():
            substitutions = [[int(l) for l in k] for k in substitutions]
            # add {'formula_name', index} to the dictionary formulas_index if not already present
            if k not in self.formulas_index:
                # TODO: check what happens with more than 1 formula in the dataset
                self.formulas_index[k] = len(self.formulas_index)

            indices_formulas_dict = {self.formulas_index[k]: []}


            # loop over all grounded rules of the form: ((head_tuple), (body_tuple_1, body_tuple_2, ...))
            for y, (head, body) in enumerate(groundings):
                indices_formulas_dict[self.formulas_index[k]].append([])


                # get head index (and check that there is only one head)
                assert len(head) == 1  # TODO: check what happens with more than 1 head
                index_head = self.unique_query_index[head[0]]

                # add {'body': index} to dictionary bodies_index if not already present
                if body not in self.bodies_index:
                    self.bodies_index[body] = len(self.bodies_index)

                # loop over all atoms in the body and generate index of the form: (body_index, formula_index, grounded_relation_index, position, head_index)
                for pos, b in enumerate(body):
                    # TODO: check whether to use grounded_relation_index or unique_atom_index or unique_query_index
                    indices_formulas.append((self.bodies_index[body], self.formulas_index[k], self.grounded_relation_index[b], pos, index_head))
                    indices_formulas_dict[self.formulas_index[k]][-1].append(self.grounded_relation_index[b])
            indices_tuples_formulas[self.formulas_index[k]] = substitutions
        return indices_formulas, indices_tuples_formulas

    def invert_index_relation(self, tuple_relation: Tuple[int, int, int, int]):
        inverted_unique_tuples = {v: k for k, v in self.unique_query_index.items()}
        inverted_relation_index = {v: k for k, v in self.relation_index.items()}
        return inverted_unique_tuples[tuple_relation[0]], \
            inverted_relation_index[tuple_relation[1]], \
            tuple_relation[2],\
            tuple_relation[3]

    def invert_index_formula(self, tuple_formula: Tuple[int, int, int, int, int]):
        inverted_unique_tuples = {v: k for k, v in self.unique_query_index.items()}
        inverted_indices_bodies = {v: k for k, v in self.bodies_index.items()}
        inverted_indices_formulas = {v: k for k, v in self.formulas_index.items()}
        return inverted_indices_bodies[tuple_formula[0]],\
            inverted_indices_formulas[tuple_formula[1]],\
            inverted_unique_tuples[tuple_formula[2]],\
            tuple_formula[3],\
            inverted_unique_tuples[tuple_formula[4]]


#
# class LogicSerializerFast():
#
#     def __init__(self, predicates: List[Predicate], domains: List[Domain]):
#         self.predicates = predicates
#         self.domains = domains
#
#         self.constant_to_global_index = defaultdict(OrderedDict)
#         for domain in domains:
#             for i, constant in enumerate(domain.constants):
#                 self.constant_to_global_index[domain.name][constant] = i
#
#         self.predicate_to_domains = {}
#         for predicate in predicates:
#             self.predicate_to_domains[predicate.name] = [domain.name for domain in predicate.domains]
#
#     def serialize(self, queries:List[Tuple], rule_groundings:List[RuleGroundings]):
#         domain_to_global = defaultdict(list)  # X_domains
#         domain_to_local_constant_index = defaultdict(dict) # helper
#         predicate_to_constant_tuples = defaultdict(list) # A_predicates
#
#         # Set of all atoms to index
#         all_atoms = list(queries)
#         # ns.utils.add_if_not_in(ns.utils.to_flat(queries), all_atoms_list, all_atoms_set)
#         for rg in rule_groundings:
#             for g in rg.groundings:
#                 all_atoms += g[0] # head
#                 all_atoms += g[1] # body
#         all_atoms = sorted(list(set(all_atoms)))
#         #        ns.utils.add_if_not_in(g[0], all_atoms_list, all_atoms_set) # head
#         #        ns.utils.add_if_not_in(g[1], all_atoms_list, all_atoms_set) # head
#         #all_atoms = all_atoms_list
#
#         # Bucket them per predicate
#         all_atoms_per_predicate = {predicate.name: [] for predicate in self.predicates}
#         for atom in all_atoms:
#             all_atoms_per_predicate[atom[0]].append(atom)
#
#         # Create the index following the bucketed order
#         atom_to_index = {}
#         for predicate in self.predicates:
#             atoms = all_atoms_per_predicate[predicate.name]
#             for atom in atoms:
#                 atom_to_index[atom] = len(atom_to_index)
#                 indices_cs = []
#                 domains = self.predicate_to_domains[atom[0]]
#
#                 for i, c in enumerate(atom[1:]):
#                     constant_index = domain_to_local_constant_index[domains[i]]
#                     if c not in constant_index:
#                         constant_index[c] = len(constant_index)
#                         domain_to_global[domains[i]].append(self.constant_to_global_index[domains[i]][c])
#                     indices_cs.append(constant_index[c])
#
#                 predicate_to_constant_tuples[atom[0]].append(indices_cs)
#
#         queries = [[atom_to_index[q] for q in Q] for Q in queries]
#         groundings = {}
#         for rule in rule_groundings:
#             if len(rule.groundings) > 0:
#                 G_body = [[atom_to_index[atom] for atom in g[1]]
#                           for g in rule.groundings]
#                 G_head = [[atom_to_index[atom] for atom in g[0]]
#                           for g in rule.groundings]
#                 groundings[rule.name] = G_body, G_head
#
#         return domain_to_global, predicate_to_constant_tuples, groundings, queries
#
#

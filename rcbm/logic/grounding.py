from typing import List, Set, Dict
from itertools import product, combinations, permutations, combinations_with_replacement
from .commons import Rule


class DomainGrounder:

    def __init__(self, domains, rules: List[Rule],
                 limit: int=0, not_reflexive: bool = True, manifolds_per_rule: Dict[str, List] = None):
        self.rules = rules
        self.domains = domains
        self.limit = limit
        self.not_reflexive = not_reflexive

        self.manifolds_per_rule = manifolds_per_rule if manifolds_per_rule is not None else {}
        for rule in rules:
            if rule.name in self.manifolds_per_rule:
                rule_arity = len(rule.vars)
                empty_tuples = set()
                for manifold in self.manifolds_per_rule[rule.name]:
                    if self.not_reflexive:
                        for t in combinations(manifold, rule_arity):
                            for p in permutations(t):
                                empty_tuples.add(p)
                    else:
                        for t in combinations_with_replacement(manifold, rule_arity):
                            for p in permutations(t):
                                empty_tuples.add(p)

                self.manifolds_per_rule[rule.name] = empty_tuples

    #@lru_cache
    def ground(self):
        res = {}
        for clause in self.rules:
            added = 0
            groundings = []
            substitutions = []

            if clause.name not in self.manifolds_per_rule:
                tuples_per_rule = product(*[self.domains[d] for d in clause.vars.values()])
            else:
                tuples_per_rule = self.manifolds_per_rule[clause.name]

            for ground_vars in tuples_per_rule:

                if self.not_reflexive and len(set(ground_vars)) < len(ground_vars):
                    continue

                var_assignments = {k:v for k,v in zip(
                    clause.vars.keys(), ground_vars)}

                # We use a lexicographical order of the variables
                constant_tuples = [v for k,v in sorted(var_assignments.items(), key= lambda x: x[0])]

                body_atoms = []
                for atom in clause.body:
                    ground_atom = (atom[0], ) + tuple(
                        [var_assignments.get(atom[j+1], None)
                         for j in range(len(atom)-1)])
                    assert all(ground_atom), 'Unresolved %s' % str(ground_atom)
                    body_atoms.append(ground_atom)

                head_atoms = []
                for atom in clause.head:
                    ground_atom = (atom[0], ) + tuple(
                        [var_assignments.get(atom[j+1], None)
                         for j in range(len(atom)-1)])
                    assert all(ground_atom), 'Unresolved %s' % str(ground_atom)
                    head_atoms.append(ground_atom)
                groundings.append((tuple(head_atoms), tuple(body_atoms)))
                substitutions.append(constant_tuples)
                added += 1
                if self.limit > 0 and self.limit >= added:
                    break

            res[clause.name] = (groundings, substitutions)
        return res
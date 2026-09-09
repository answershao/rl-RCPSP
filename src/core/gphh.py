"""Genetic-programming hyper-heuristic (GPHH) for single-project RCPSP.

A priority rule is represented as an arithmetic expression tree over the static
per-activity features of :func:`src.core.rules.activity_features` (duration,
CPM time windows, slack, transitive successor count/duration, resource demand).
Evaluating the tree on an activity's feature vector yields that activity's
priority score; higher scores are scheduled earlier, which is exactly the score
convention of :func:`src.core.rcpsp.generate_schedule`.  An evolved rule is
therefore a single-pass priority rule directly comparable with FIFO/SPT/LST/...

Trees are nested tuples.  A leaf is either a feature name (``str``) or a float
constant; an internal node is ``(op, left, right)`` with ``op`` in
``FUNCTIONS``.  Fitness of one rule on one instance is the relative deviation
from the CPM (precedence-only) lower bound, ``(makespan - LB) / LB``; the
per-rule fitness on a training set is the mean over instances, which keeps
instances of different difficulty comparable.

Evolution follows a canonical GP loop: ramped half-and-half initialisation,
tournament selection with elitism, subtree crossover, and subtree mutation
(protected division and finite-score sanitation keep every decode valid).
All randomness flows through a single ``random.Random(seed)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from src.core.rules import FEATURE_NAMES, activity_features
from src.core.rcpsp import Instance, generate_schedule

# op name -> (arity, function).  All functions are binary here.
FUNCTIONS: dict[str, tuple[int, object]] = {
    "add": (2, lambda a, b: a + b),
    "sub": (2, lambda a, b: a - b),
    "mul": (2, lambda a, b: a * b),
    "div": (2, lambda a, b: a / b if abs(b) > 1e-9 else 1.0),
    "max": (2, max),
    "min": (2, min),
}

# Leaf constants mixed into the terminal set (float leaves).
CONSTANTS = (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0)

_MAX_SCORE = 1e12  # sanitation bound for non-finite rule outputs

DEFAULT_POPULATION = 100
DEFAULT_GENERATIONS = 30
DEFAULT_TOURNAMENT = 5
DEFAULT_ELITE = 2
DEFAULT_MAX_DEPTH = 6
DEFAULT_CROSSOVER_PROB = 0.9
DEFAULT_MUTATION_PROB = 0.1


@dataclass(frozen=True)
class GPHHParameters:
    population: int = DEFAULT_POPULATION
    generations: int = DEFAULT_GENERATIONS
    tournament: int = DEFAULT_TOURNAMENT
    elite: int = DEFAULT_ELITE
    max_depth: int = DEFAULT_MAX_DEPTH
    crossover_prob: float = DEFAULT_CROSSOVER_PROB
    mutation_prob: float = DEFAULT_MUTATION_PROB

    def __post_init__(self) -> None:
        if self.population < 2:
            raise ValueError("population must be at least 2")
        if self.generations < 1:
            raise ValueError("generations must be positive")
        if self.tournament < 1:
            raise ValueError("tournament must be positive")
        if not 0 <= self.elite < self.population:
            raise ValueError("elite must be in [0, population)")
        if self.max_depth < 2:
            raise ValueError("max_depth must be at least 2")
        if not 0 <= self.crossover_prob <= 1 or not 0 <= self.mutation_prob <= 1:
            raise ValueError("crossover_prob and mutation_prob must be in [0, 1]")


@dataclass(frozen=True)
class GPHHResult:
    best_rule: object  # expression-tree root
    best_fitness: float  # mean relative deviation from CPM lower bound on training set
    best_generation: int
    history: tuple[float, ...]  # best training fitness at the end of each generation
    evaluations: int
    instance_count: int  # size of the training set


# ---------------------------------------------------------------------------
# Tree primitives
# ---------------------------------------------------------------------------

def is_terminal(node: object) -> bool:
    return isinstance(node, (str, int, float))


def depth(node: object) -> int:
    """Depth of a tree (a leaf has depth 1)."""
    if is_terminal(node):
        return 1
    return 1 + max(depth(child) for child in node[1:])


def size(node: object) -> int:
    """Number of nodes in a tree."""
    if is_terminal(node):
        return 1
    return 1 + sum(size(child) for child in node[1:])


def random_terminal(rng: random.Random) -> object:
    if rng.random() < 0.5:
        return rng.choice(FEATURE_NAMES)
    return rng.choice(CONSTANTS)


def _random_tree(rng: random.Random, depth_limit: int, *, full: bool) -> object:
    """Koza-style grow/full subtree of at most ``depth_limit`` levels."""
    if depth_limit <= 1:
        return random_terminal(rng)
    if full or rng.random() < 0.3:
        op = rng.choice(tuple(FUNCTIONS))
        arity = FUNCTIONS[op][0]
        return (op,) + tuple(_random_tree(rng, depth_limit - 1, full=full) for _ in range(arity))
    return random_terminal(rng)


def random_tree(rng: random.Random, depth_limit: int, *, full: bool = False) -> object:
    return _random_tree(rng, depth_limit, full=full)


def _iter_paths(node: object, prefix: tuple[int, ...] = ()):
    """Yield ``(path, subtree)`` for every node, pre-order."""
    yield prefix, node
    if not is_terminal(node):
        for index, child in enumerate(node[1:]):
            yield from _iter_paths(child, prefix + (index,))


def _replace(node: object, path: tuple[int, ...], new: object) -> object:
    """Return a copy of ``node`` with the subtree at ``path`` replaced."""
    if not path:
        return new
    head, *rest = path
    children = list(node[1:])
    children[head] = _replace(children[head], tuple(rest), new)
    return (node[0],) + tuple(children)


def _random_node(rng: random.Random, node: object) -> tuple[tuple[int, ...], object]:
    paths = list(_iter_paths(node))
    return paths[rng.randrange(len(paths))]


def crossover(
    rng: random.Random, a: object, b: object, max_depth: int
) -> object:
    """Subtree crossover; returns a child no deeper than ``max_depth``.

    If swapping the chosen subtrees would exceed ``max_depth`` the mutation is
    retried a few times on shallower points before falling back to a clone of
    ``a`` (standard depth-guard behaviour).
    """
    for _ in range(20):
        path_a, subtree_a = _random_node(rng, a)
        path_b, subtree_b = _random_node(rng, b)
        child_a = _replace(a, path_a, subtree_b)
        child_b = _replace(b, path_b, subtree_a)
        if depth(child_a) <= max_depth:
            return child_a
        if depth(child_b) <= max_depth:
            return child_b
    return a


def mutate(rng: random.Random, rule: object, max_depth: int) -> object:
    """Subtree mutation: replace a random subtree with a fresh random one.

    A subtree at a path of length ``k`` may grow at most ``max_depth - k`` new
    levels, otherwise the whole tree would exceed ``max_depth``.
    """
    path, _ = _random_node(rng, rule)
    limit = max_depth - len(path)
    if limit < 1:
        return rule
    fresh = random_tree(rng, limit, full=False)
    child = _replace(rule, path, fresh)
    return child if depth(child) <= max_depth else rule


def evaluate_tree(rule: object, features: dict[str, float]) -> float:
    """Priority score of one activity (higher = earlier)."""
    if isinstance(rule, str):
        value = float(features[rule])
    elif isinstance(rule, (int, float)):
        value = float(rule)
    else:
        op = rule[0]
        _, function = FUNCTIONS[op]
        value = function(*(evaluate_tree(child, features) for child in rule[1:]))
    if math.isnan(value):
        return 0.0
    if math.isinf(value):
        return _MAX_SCORE if value > 0 else -_MAX_SCORE
    return value


def rule_priorities(
    instance: Instance, rule: object
) -> dict[int, float]:
    """Score map (higher = earlier) produced by an evolved rule on one instance."""
    features = activity_features(instance)
    return {
        aid: evaluate_tree(rule, values)
        for aid, values in features.items()
    }


def cpm_lower_bound(instance: Instance) -> int:
    """Precedence-only makespan lower bound (max CPM earliest finish)."""
    features = activity_features(instance)
    return max(0, int(max(values["eft"] for values in features.values())))


def rule_makespan(instance: Instance, rule: object) -> int:
    """Serial-SGS makespan of an evolved rule on one instance (validated)."""
    return generate_schedule(instance, rule_priorities(instance, rule)).makespan


def rule_relative_deviation(instance: Instance, rule: object) -> float:
    """(makespan - CPM-LB) / CPM-LB of one evolved rule on one instance."""
    lower_bound = cpm_lower_bound(instance)
    makespan = rule_makespan(instance, rule)
    if lower_bound <= 0:
        return float(makespan)
    return (makespan - lower_bound) / lower_bound


def rule_to_string(rule: object) -> str:
    """Readable infix rendering, e.g. ``(lst - est) + (dur * 1)``."""
    if isinstance(rule, str):
        return rule
    if isinstance(rule, (int, float)):
        return f"{float(rule):g}"
    op = rule[0]
    if op in ("add", "sub", "mul", "div"):
        symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
        return f"({rule_to_string(rule[1])} {symbol} {rule_to_string(rule[2])})"
    return f"{op}({rule_to_string(rule[1])}, {rule_to_string(rule[2])})"


# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------

def _fitness_on_instances(rule: object, instances: list[Instance]) -> float:
    if not instances:
        raise ValueError("training instance set must not be empty")
    total = 0.0
    for instance in instances:
        total += rule_relative_deviation(instance, rule)
    return total / len(instances)


def run_gphh(
    instances: list[Instance],
    *,
    seed: int,
    parameters: GPHHParameters | None = None,
) -> GPHHResult:
    """Evolve a priority rule on ``instances`` with a canonical GP loop."""
    parameters = parameters or GPHHParameters()
    rng = random.Random(seed)
    population = parameters.population
    generations = parameters.generations

    def make_rule() -> object:
        depth_limit = rng.randint(2, parameters.max_depth)
        return random_tree(rng, depth_limit, full=rng.random() < 0.5)

    rules = [make_rule() for _ in range(population)]
    cache: dict[str, float] = {}
    fitness: list[float] = []

    def evaluate(rule: object) -> float:
        key = repr(rule)
        if key not in cache:
            cache[key] = _fitness_on_instances(rule, instances)
        return cache[key]

    fitness = [evaluate(rule) for rule in rules]
    evaluations = len(cache)

    def tournament_best() -> int:
        size = min(parameters.tournament, len(rules))
        candidates = rng.sample(range(len(rules)), size)
        return min(candidates, key=lambda index: fitness[index])

    best_index = min(range(len(rules)), key=lambda index: fitness[index])
    best_rule = rules[best_index]
    best_fitness = fitness[best_index]
    best_generation = 0
    history: list[float] = [best_fitness]

    for generation in range(1, generations + 1):
        ranked = sorted(range(len(rules)), key=lambda index: fitness[index])
        next_rules: list[object] = [rules[index] for index in ranked[: parameters.elite]]
        while len(next_rules) < population:
            parent_a = tournament_best()
            parent_b = tournament_best()
            roll = rng.random()
            if roll < parameters.crossover_prob:
                child = crossover(rng, rules[parent_a], rules[parent_b], parameters.max_depth)
            elif roll < parameters.crossover_prob + parameters.mutation_prob:
                child = mutate(rng, rules[parent_a], parameters.max_depth)
            else:
                child = rules[parent_a]
            next_rules.append(child)
        rules = next_rules
        fitness = [evaluate(rule) for rule in rules]
        evaluations = len(cache)

        generation_best = min(range(len(rules)), key=lambda index: fitness[index])
        if fitness[generation_best] < best_fitness:
            best_fitness = fitness[generation_best]
            best_rule = rules[generation_best]
            best_generation = generation
        history.append(best_fitness)

    return GPHHResult(
        best_rule=best_rule,
        best_fitness=best_fitness,
        best_generation=best_generation,
        history=tuple(history),
        evaluations=evaluations,
        instance_count=len(instances),
    )

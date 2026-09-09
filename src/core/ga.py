"""Random-key genetic algorithm baseline for single-project RCPSP.

Every chromosome is a vector of ``n_activities`` keys in ``[0, 1)`` (one per
activity, in ascending activity-id order).  Decoding maps keys to the priority
score map consumed by the serial SGS (:func:`src.core.rcpsp.generate_schedule`)
-- higher key = scheduled earlier -- which is exactly the same decoder used by
the deterministic priority rules in :mod:`src.core.rules`.  Fitness is the
makespan (minimised).

Operators are classic random-key GA operators:
  * tournament selection (``tournament`` participants, elitist replacement);
  * uniform crossover (each gene inherited from either parent with p=0.5);
  * per-gene mutation that resamples the key from U[0, 1) with p=``mutation``
    (defaults to ``1 / n_activities``);
  * elitism keeps the best ``elite`` individuals unchanged each generation.

Everything is driven by a single ``random.Random(seed)`` so a run is exactly
reproducible for a fixed seed, instance, and set of hyper-parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from src.core.rcpsp import ActivityId, Instance, generate_schedule

# Number of solution evaluations is population * generations.
DEFAULT_POPULATION = 50
DEFAULT_GENERATIONS = 200
DEFAULT_TOURNAMENT = 3
DEFAULT_ELITE = 2


@dataclass(frozen=True)
class GAParameters:
    """Hyper-parameters for one GA run."""

    population: int = DEFAULT_POPULATION
    generations: int = DEFAULT_GENERATIONS
    tournament: int = DEFAULT_TOURNAMENT
    elite: int = DEFAULT_ELITE
    # ``None`` means the per-gene default ``1 / n_activities``.
    mutation: float | None = None

    def __post_init__(self) -> None:
        if self.population < 2:
            raise ValueError("population must be at least 2")
        if self.generations < 1:
            raise ValueError("generations must be positive")
        if self.tournament < 1:
            raise ValueError("tournament must be positive")
        if not 0 <= self.elite < self.population:
            raise ValueError("elite must be in [0, population)")
        if self.mutation is not None and not 0 <= self.mutation <= 1:
            raise ValueError("mutation must be in [0, 1]")


@dataclass(frozen=True)
class GAResult:
    """Outcome of one GA run."""

    best_makespan: int
    best_keys: tuple[float, ...]
    best_generation: int
    history: tuple[int, ...]  # best makespan seen at the end of each generation
    evaluations: int


def decode_keys(instance: Instance, keys: tuple[float, ...] | list[float]) -> dict[ActivityId, float]:
    """Map a chromosome to the ``{activity: priority}`` score map for the serial SGS."""
    if len(keys) != len(instance.activities):
        raise ValueError(
            f"chromosome length {len(keys)} does not match activity count {len(instance.activities)}"
        )
    # Ascending activity-id order is the convention used by the environment.
    ids = sorted(instance.activities)
    return {activity_id: float(keys[index]) for index, activity_id in enumerate(ids)}


def evaluate(instance: Instance, keys: tuple[float, ...] | list[float]) -> int:
    """Feasible serial-SGS makespan of one chromosome (always validated)."""
    return generate_schedule(instance, decode_keys(instance, keys)).makespan


def run_ga(
    instance: Instance,
    *,
    seed: int,
    parameters: GAParameters | None = None,
) -> GAResult:
    """Run the random-key GA and return the best solution found."""
    parameters = parameters or GAParameters()
    rng = random.Random(seed)
    n = len(instance.activities)
    mutation = parameters.mutation if parameters.mutation is not None else 1.0 / n
    population = parameters.population
    generations = parameters.generations

    # population: list of chromosome tuples (immutable, dict/hash friendly).
    keys_pop: list[tuple[float, ...]] = [
        tuple(rng.random() for _ in range(n)) for _ in range(population)
    ]

    def makespan_of(keys: tuple[float, ...]) -> int:
        return generate_schedule(instance, decode_keys(instance, keys)).makespan

    fitness = [makespan_of(keys) for keys in keys_pop]

    best_index = min(range(population), key=lambda index: fitness[index])
    best_keys = keys_pop[best_index]
    best_makespan = fitness[best_index]
    best_generation = 0
    history: list[int] = [best_makespan]
    evaluations = population

    for generation in range(1, generations + 1):
        # Sort indices by fitness ascending (minimisation); stable keeps ties in
        # the same chromosome order so elitism is deterministic.
        ranked = sorted(range(population), key=lambda index: fitness[index])
        next_pop: list[tuple[float, ...]] = [
            keys_pop[index] for index in ranked[: parameters.elite]
        ]

        while len(next_pop) < population:
            parent_a = _tournament_parent(rng, ranked, fitness, parameters.tournament)
            parent_b = _tournament_parent(rng, ranked, fitness, parameters.tournament)
            child = tuple(
                keys_pop[parent_a][gene] if rng.random() < 0.5 else keys_pop[parent_b][gene]
                for gene in range(n)
            )
            if mutation > 0:
                child = tuple(
                    rng.random() if rng.random() < mutation else value for value in child
                )
            next_pop.append(child)

        keys_pop = next_pop
        fitness = [makespan_of(keys) for keys in keys_pop]
        evaluations += population

        generation_best = min(range(population), key=lambda index: fitness[index])
        if fitness[generation_best] < best_makespan:
            best_makespan = fitness[generation_best]
            best_keys = keys_pop[generation_best]
            best_generation = generation
        history.append(best_makespan)

    return GAResult(
        best_makespan=best_makespan,
        best_keys=best_keys,
        best_generation=best_generation,
        history=tuple(history),
        evaluations=evaluations,
    )


def _tournament_parent(
    rng: random.Random,
    ranked: list[int],
    fitness: list[int],
    tournament: int,
) -> int:
    """Return the index of the tournament winner (best among sampled)."""
    candidates = rng.sample(ranked, min(tournament, len(ranked)))
    return min(candidates, key=lambda index: fitness[index])

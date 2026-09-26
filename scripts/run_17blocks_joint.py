"""Run one joint TPE services optimization for a set of target blocks.

All target blocks are optimized in one Facade/TPE problem; this is not a loop
of independent per-block optimizations.
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Seed-fix: TPE/Optuna trials are otherwise non-deterministic. The "every-5th
# explore" patch in optimizer.py only helps if runs are reproducible; without
# a seed, a single best=0.677 run is indistinguishable from RNG luck.
_DEFAULT_SEED = int(os.environ.get("TPE_SEED", "0"))
np.random.seed(_DEFAULT_SEED)
try:
    optuna_seed = int(os.environ.get("OPTUNA_SEED", str(_DEFAULT_SEED)))
except ValueError:
    optuna_seed = _DEFAULT_SEED

from blocksnet.config import service_types_config
from blocksnet.enums import LandUse
from blocksnet.optimization.services import (
    AreaSolution,
    Facade,
    GradientChooser,
    RandomOrder,
    TPEOptimizer,
    WeightedConstraints,
    WeightedObjective,
)


DEFAULT_BLOCK_IDS = [
    823,
    955,
    1071,
    1129,
    1182,
    1411,
    1428,
    1429,
    1433,
    1473,
    1518,
    1526,
    1907,
    2043,
    2093,
    2098,
    2886,
]

# Priority tiers and their weights, as in examples/optimization/services/tpe_simple.ipynb.
# Tiers are normalised 4/7, 2/7, 1/7; the weight inside a tier scales that share.
BASIC_SERVICES = {
    "kindergarten": 0.2,
    "school": 0.2,
    "pharmacy": 0.2,
    "polyclinic": 0.3,
    "convenience": 0.2,
    "cafe": 0.15,
    "playground": 0.2,
    "post": 0.1,
    "hairdresser": 0.1,
}
ADVANCED_SERVICES = {
    "hospital": 0.5,
    "supermarket": 0.4,
    "restaurant": 0.3,
    "park": 0.8,
    "bank": 0.3,
    "fuel": 0.25,
    "police": 0.2,
    "cinema": 0.2,
    "market": 0.2,
    "religion": 0.5,
    "multifunctional_center": 0.1,
    "train_building": 0.1,
    "cemetery": 0.1,
    "notary": 0.1,
    "pitch": 0.2,
}
COMFORT_SERVICES = {
    "mall": 0.3,
    "swimming_pool": 0.4,
    "theatre": 0.2,
    "museum": 0.2,
    "university": 0.2,
    "sanatorium": 0.2,
    "zoo": 0.1,
    "reserve": 0.1,
    "animal_shelter": 0.1,
    "aeroway_terminal": 0.2,
    "crematorium": 0.1,
    "prison": 0.1,
    "landfill": 0.1,
    "bar": 0.3,
    "hotel": 0.3,
}
# Each tier occupies a band of its own, so a basic service always outweighs an
# advanced one and an advanced one a comfort one, whatever their weights inside
# the tier. Multiplying a tier share by a weight that runs from 0.1 to 0.8
# instead let the bands overlap: a hospital scored 0.5 * 0.2857 = 0.143 against
# a kindergarten's 0.2 * 0.5714 = 0.114 and was served first.
TIER_BANDS = (
    (BASIC_SERVICES, 0.40, 0.57),
    (ADVANCED_SERVICES, 0.20, 0.29),
    (COMFORT_SERVICES, 0.05, 0.14),
)
# A service the tiers do not mention is treated as comfort at its lowest.
UNLISTED_WEIGHT = 0.05


def capacity_column(service_type: str) -> str:
    return f"capacity_{service_type}"


def get_service_weight(service_type: str) -> float:
    """Weight of a service type: its tier's band, its own place inside it.

    Parameters
    ----------
    service_type : str
        Service type identifier.

    Returns
    -------
    float
        Weight in the objective. Tiers never overlap, so the basic services are
        always the heaviest and are filled first.
    """
    for tier, low, high in TIER_BANDS:
        if service_type in tier:
            weights = list(tier.values())
            span = max(weights) - min(weights)
            share = (tier[service_type] - min(weights)) / span if span else 1.0
            return round(low + (high - low) * share, 4)
    return UNLISTED_WEIGHT


def get_service_weights(blocks: pd.DataFrame, land_use: LandUse) -> dict[str, float]:
    """Every service type the land use allows and the data can supply.

    The catalogue comes from ``service_types_config`` rather than a list in this
    file: a hand-kept list silently drops what it misspells, which is how
    ``polyclinics`` (the config spells it ``polyclinic``) and ``post_office``
    (``post``) disappeared from earlier runs without a word.
    """
    service_types = sorted(
        service_type
        for service_type in service_types_config[land_use]
        if capacity_column(service_type) in blocks.columns
    )
    return {service_type: get_service_weight(service_type) for service_type in service_types}


def load_data(project_root: Path) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    blocks = gpd.read_file(project_root / "data/blocks_with_buildings_and_services.gpkg")
    acc_mx = pd.read_pickle(project_root / os.environ.get("ACC_MX", "data/acc_mx.pickle"))
    # site_area is the block's own land: the area of its polygon. It is what
    # FSI and GSI are taken against, and what both housing and service plots
    # are carved out of. footprint_area is a different thing — the area the
    # buildings cover — and standing in for site_area it understated every
    # block by the inverse of its GSI (block 823: 10869 m2 instead of 57549).
    if "site_area" not in blocks.columns:
        blocks = blocks.copy()
        blocks["site_area"] = blocks.geometry.area
    blocks = blocks.drop(columns=["geometry"]).copy()
    keep = blocks.index
    acc_mx = acc_mx.loc[acc_mx.index.intersection(keep), acc_mx.columns.intersection(keep)]
    return blocks, acc_mx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-ids", nargs="+", type=int, default=DEFAULT_BLOCK_IDS)
    parser.add_argument("--max-runs", type=int, default=120)
    parser.add_argument("--timeout", type=float, default=600.0)
    # Left unset, the budget is derived from the catalogue: see resolve_max_evals.
    parser.add_argument("--max-evals", type=int, default=None)
    parser.add_argument("--n-ei-candidates", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/optimization/services/17blocks_joint_optimization"),
    )
    return parser.parse_args()


def resolve_max_evals(max_evals: int | None, max_runs: int, service_count: int) -> int:
    """Budget of provision computations the optimizer may spend.

    ``WeightedObjective`` counts this budget in recomputed provisions, not in
    trials: a trial spends one for every service type it changed. So the budget
    that buys N trials depends on how many service types are in play, and a
    fixed number silently shortens the search as the catalogue grows. With the
    nine-type profile 200 bought about 28 trials; with the full catalogue of 33
    it bought 8, which is why three quarters of the variables were never
    touched and the same few service types came out of every run.

    Derived here as "every trial may touch every service type", so the trial
    count stays what ``--max-runs`` says. The run is still bounded by
    ``--max-runs`` and ``--timeout``.

    Parameters
    ----------
    max_evals : int | None
        Explicit budget, or None to derive one.
    max_runs : int
        Maximum number of trials.
    service_count : int
        Number of service types the run actually optimizes.

    Returns
    -------
    int
        Budget of provision computations.
    """
    if max_evals is not None:
        return max_evals
    return max_runs * max(service_count, 1)


def write_coverage_report(
    solution_services: pd.DataFrame, population: int, service_types: dict, path: Path
) -> pd.DataFrame:
    """Compare what the plan provides with what the norms require.

    The run used to report only counts and areas, and a plan that covered none
    of the demand for kindergartens, schools or pharmacies looked unremarkable
    among them. Coverage makes that visible in the artifacts of the run itself.

    Parameters
    ----------
    solution_services : pd.DataFrame
        Services placed by the solution.
    population : int
        Residents the blocks house under that solution.
    service_types : dict
        Service types the run optimized, mapped to their weights.
    path : Path
        Where to write the report.

    Returns
    -------
    pd.DataFrame
        The report, worst coverage first.
    """
    provided = (
        (solution_services["capacity"] * solution_services["count"]).groupby(solution_services["service_type"]).sum()
        if len(solution_services)
        else pd.Series(dtype=float)
    )
    rows = []
    for service_type in sorted(service_types):
        _, demand, _ = service_types_config[service_type].values()
        required = population / 1000 * demand
        have = float(provided.get(service_type, 0.0))
        rows.append(
            {
                "service_type": service_type,
                "name_ru": service_types_config[service_type]["name_ru"],
                "required_capacity": round(required, 1),
                "provided_capacity": have,
                "coverage": round(have / required, 4) if required > 0 else None,
            }
        )
    report = pd.DataFrame(rows).sort_values("coverage", na_position="last")
    report.to_csv(path, index=False)
    return report


def verify_solution(facade: Facade, variables: list, solution_services: pd.DataFrame) -> None:
    """Fail loudly if the reported plan is not the plan that was optimized.

    The converter carries the previous trial's solution on every other call, so
    reporting through it once described the same ``best_x`` as 169 services and
    once as 102. Both checks below would have caught that the day it appeared.

    Parameters
    ----------
    facade : Facade
        Facade the solution came from.
    variables : list
        The placed services, as the best trial recorded them.
    solution_services : pd.DataFrame
        Services reported for it.

    Raises
    ------
    AssertionError
        If the plan breaks the constraints or does not match what is reported.
    """
    assert facade.check_variables(variables), "reported plan violates the constraints"
    placed = sum(variable.count for variable in variables)
    reported = 0 if solution_services.empty else int(solution_services["count"].sum())
    assert placed == reported, f"reported plan does not match the trial: {placed} placed, {reported} reported"


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = (project_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    block_ids = list(dict.fromkeys(int(block_id) for block_id in args.block_ids))
    blocks, acc_mx = load_data(project_root)
    missing_blocks = [block_id for block_id in block_ids if block_id not in blocks.index]
    missing_acc = [block_id for block_id in block_ids if block_id not in acc_mx.index or block_id not in acc_mx.columns]
    if missing_blocks or missing_acc:
        raise ValueError(f"Invalid inputs: missing_blocks={missing_blocks}, missing_acc={missing_acc}")

    land_use = LandUse.RESIDENTIAL
    service_weights = get_service_weights(blocks, land_use)
    if not service_weights:
        raise ValueError("No configured service capacity columns are available")

    blocks_lu = {block_id: land_use for block_id in block_ids}
    facade = Facade(
        var_adapter=AreaSolution(blocks_lu),
        accessibility_matrix=acc_mx,
        blocks_df=blocks,
        blocks_lu=blocks_lu,
    )
    for service_type, weight in service_weights.items():
        column = capacity_column(service_type)
        services_df = blocks[[column]].rename(columns={column: "capacity"})
        facade.add_service_type(service_type, weight, services_df)
    active_service_weights = {
        service_type: service_weights[service_type]
        for service_type in sorted(facade.start_provisions)
    }
    skipped_service_types = sorted(set(service_weights) - set(active_service_weights))

    max_evals = resolve_max_evals(args.max_evals, args.max_runs, len(active_service_weights))
    objective = WeightedObjective(
        num_params=facade.num_params,
        facade=facade,
        weights=service_weights,
        max_evals=max_evals,
    )
    constraints = WeightedConstraints(
        num_params=facade.num_params,
        facade=facade,
        priority=active_service_weights,
    )
    optimizer = TPEOptimizer(
        objective=objective,
        constraints=constraints,
        vars_order=RandomOrder(),
        vars_chooser=GradientChooser(
            facade, num_params=facade.num_params, num_top=5, weights=active_service_weights
        ),
        n_ei_candidates=args.n_ei_candidates,
        sampler_seed=optuna_seed,
    )

    input_columns = [
        column
        for column in (
            "population",
            "site_area",
            "footprint_area",
            "build_floor_area",
            "living_area",
            "non_living_area",
        )
        if column in blocks.columns
    ]
    blocks.loc[block_ids, input_columns].rename_axis("block_id").to_csv(output_dir / "input_blocks.csv")

    run_config = {
        "block_ids": block_ids,
        "joint_optimization": True,
        "land_use": "RESIDENTIAL",
        "service_weights": service_weights,
        "active_service_weights": active_service_weights,
        "skipped_service_types": skipped_service_types,
        "max_runs": args.max_runs,
        "timeout_sec": args.timeout,
        "max_evals": max_evals,
        "n_ei_candidates": args.n_ei_candidates,
        "num_params": facade.num_params,
        "data_files": [
            "data/blocks_with_buildings_and_services.gpkg",
            "data/acc_mx.pickle",
        ],
        "site_area_source": "site_area column, otherwise the block polygon area",
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(
        f"Joint optimization: blocks={len(block_ids)} services={len(active_service_weights)} "
        f"params={facade.num_params} max_runs={args.max_runs} timeout={args.timeout}s",
        flush=True,
    )
    started = time.time()
    original_cwd = Path.cwd()
    try:
        os.chdir(output_dir)
        best_x, best_val, success_rate, func_evals = optimizer.run(
            max_runs=args.max_runs,
            timeout=args.timeout,
            initial_runs_num=1,
            verbose=False,
        )
    finally:
        os.chdir(original_cwd)
    elapsed = time.time() - started

    # The plan is the counts the best trial ended up with, not a re-reading of
    # its solution vector: the converter interprets a vector against the
    # previous trial, so re-reading it yields a different, possibly infeasible
    # plan. Every trial records what it actually placed.
    best_counts = optimizer._study.best_trial.user_attrs.get("plan")
    if best_counts is None:
        solution_services = facade.solution_to_services_df(best_x)
        best_variables = facade.variables_from_counts([row.count for row in facade._converter._X])
    else:
        solution_services = facade.services_df_from_counts(best_counts)
        best_variables = facade.variables_from_counts(best_counts)
    verify_solution(facade, best_variables, solution_services)
    (output_dir / "best_x.json").write_text(json.dumps(best_x, indent=2), encoding="utf-8")
    solution_services.to_csv(output_dir / "solution_services.csv", index=False)
    solution_area = facade.get_solution_area_df(best_x).loc[block_ids]
    solution_area.rename_axis("block_id").to_csv(output_dir / "solution_area.csv")

    if solution_services.empty:
        per_block = pd.DataFrame(index=pd.Index(block_ids, name="block_id"))
        per_block["num_units"] = 0
        per_block["service_types_used"] = 0
        per_block["total_capacity"] = 0
        per_block["service_site_area"] = 0.0
        per_block["service_build_floor_area"] = 0.0
    else:
        allocated = solution_services.copy()
        allocated["total_capacity"] = allocated["capacity"] * allocated["count"]
        allocated["total_site_area"] = allocated["site_area"] * allocated["count"]
        allocated["total_build_floor_area"] = allocated["build_floor_area"] * allocated["count"]
        per_block = allocated.groupby("block_id").agg(
            num_units=("count", "sum"),
            service_types_used=("service_type", "nunique"),
            total_capacity=("total_capacity", "sum"),
            service_site_area=("total_site_area", "sum"),
            service_build_floor_area=("total_build_floor_area", "sum"),
        )
        per_block = per_block.reindex(block_ids, fill_value=0)
        per_block.index.name = "block_id"
    per_block.join(solution_area, rsuffix="_reported").to_csv(output_dir / "per_block_summary.csv")

    trial_rows = []
    for trial in optimizer._study.trials:
        trial_rows.append(
            {
                "trial_number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "suggested": trial.user_attrs.get("suggested"),
                "provisions": json.dumps(trial.user_attrs.get("provisions"), sort_keys=True),
            }
        )
    pd.DataFrame(trial_rows).to_csv(output_dir / "trials_summary.csv", index=False)

    used_bfa = solution_services.assign(
        total=solution_services["build_floor_area"] * solution_services["count"]
    ).groupby("block_id")["total"].sum() if not solution_services.empty else {}
    population = sum(
        facade._block_population(block_id, float(used_bfa.get(block_id, 0.0))) for block_id in block_ids
    )
    coverage = write_coverage_report(
        solution_services, population, active_service_weights, output_dir / "coverage.csv"
    )
    uncovered = coverage[coverage["coverage"].fillna(0) == 0]["service_type"].tolist()

    best_trial = optimizer._study.best_trial
    summary = {
        **run_config,
        "status": "ok",
        "elapsed_sec": elapsed,
        "best_value": float(best_val),
        "success_rate": float(success_rate),
        "func_evals": int(func_evals),
        "completed_trials": sum(trial.state.name == "COMPLETE" for trial in optimizer._study.trials),
        "pruned_trials": sum(trial.state.name == "PRUNED" for trial in optimizer._study.trials),
        "best_trial_number": best_trial.number,
        "best_provisions": best_trial.user_attrs.get("provisions", {}),
        "population": int(population),
        "max_population": int(facade.max_population),
        "uncovered_service_types": uncovered,
        "solution_rows": len(solution_services),
        "solution_units": int(solution_services["count"].sum()) if not solution_services.empty else 0,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise

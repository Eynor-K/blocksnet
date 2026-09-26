"""Regression tests for service over-allocation (2026-08-28).

Three defects let the optimizer propose service counts a block cannot hold and
that nobody in reach needs:

* built-in units (``site_area == 0``) were expanded without being charged to
  the area granted by the solution vector, so they appeared even for a zero
  solution and filled the whole build floor area budget of a block;
* demand is a catchment quantity but was checked per block, so every block in
  a catchment could satisfy that catchment's demand on its own;
* the objective scored provision only. Services both add capacity and displace
  the residents whose demand they would have had to meet, so housing nobody
  was always at least as good as housing people.
"""

import numpy as np
import pandas as pd
import pytest

from blocksnet.enums import LandUse
from blocksnet.optimization.services.acl.checkers.capacity_checker import CapacityChecker
from blocksnet.optimization.services.acl.variable_adapters.area_solution import AreaSolution, Unit, unit_area
from blocksnet.optimization.services.common.variable import Variable
from blocksnet.optimization.services.core.objective import WeightedObjective


BLOCK_ID = 1
SERVICE = "kindergarten"


# --------------------------------------------------------------- built-in units


def _built_in_adapter(monkeypatch):
    """AreaSolution over one block whose only unit type is built-in."""
    import blocksnet.optimization.services.acl.variable_adapters.area_solution as mod

    units_df = pd.DataFrame([{"service_type": SERVICE, "site_area": 0.0, "build_floor_area": 100.0, "capacity": 50}])

    class _StubConfig:
        units = units_df

        def __getitem__(self, _key):
            return {SERVICE: None}

    monkeypatch.setattr(mod, "service_types_config", _StubConfig())

    adapter = AreaSolution({BLOCK_ID: LandUse.RESIDENTIAL})
    adapter.add_service_type_vars(SERVICE)
    adapter.update_bfa({BLOCK_ID: 10_000.0})
    adapter.update_gfa({BLOCK_ID: 10_000.0})
    adapter.update_cap({BLOCK_ID: {SERVICE: 10_000.0}})
    return adapter


def test_unit_area_meters_built_in_units_by_floor_area():
    assert unit_area(Unit(site_area=0.0, build_floor_area=100.0, capacity=50)) == 100.0
    assert unit_area(Unit(site_area=800.0, build_floor_area=70.0, capacity=70)) == 800.0


def test_zero_solution_places_no_built_in_services(monkeypatch):
    """Granting no area must place nothing.

    Built-in units used to ignore the granted area and be capped only by the
    block's build floor area, so ``x = 0`` still produced 100 units here.
    """
    adapter = _built_in_adapter(monkeypatch)
    assert sum(var.count for var in adapter(np.zeros(len(adapter.X)))) == 0


def test_built_in_services_are_capped_by_granted_area(monkeypatch):
    """The granted area bounds the count, not the block's whole BFA budget."""
    adapter = _built_in_adapter(monkeypatch)
    assert sum(var.count for var in adapter(np.array([350]))) == 3


def test_area_variable_charges_both_area_budgets(monkeypatch):
    adapter = _built_in_adapter(monkeypatch)
    area_var = adapter.X[0]
    assert area_var.site_area == 1
    assert area_var.build_floor_area == 1
    assert area_var.capacity == pytest.approx(0.5)  # 50 capacity per 100 m² of the leading unit


# ------------------------------------------------------------- shared catchment


def _capacity_checker(monkeypatch, accessibility_matrix, demand_left, neighbourhood_share=1.0):
    import blocksnet.optimization.services.acl.checkers.capacity_checker as mod

    class _StubConfig:
        units = pd.DataFrame([{"service_type": SERVICE, "capacity": 10}])

        def __getitem__(self, _key):
            return {"name_ru": SERVICE, "demand": 1, "accessibility": 15}

    monkeypatch.setattr(mod, "service_types_config", _StubConfig())
    monkeypatch.setattr(mod, "DEMAND_LEFT_COLUMN", "demand_left")

    checker = CapacityChecker(list(accessibility_matrix.index), accessibility_matrix, neighbourhood_share)
    checker.add_service_type(SERVICE, pd.DataFrame({"demand_left": demand_left}, index=accessibility_matrix.index))
    return checker


def _placed(count_1, count_2):
    return [
        Variable(block_id=1, service_type=SERVICE, capacity=10, site_area=0, build_floor_area=1, count=count_1),
        Variable(block_id=2, service_type=SERVICE, capacity=10, site_area=0, build_floor_area=1, count=count_2),
    ]


def test_demand_is_shared_across_the_catchment(monkeypatch):
    """Blocks inside one catchment compete for the same demand.

    Blocks 1 and 2 are 5 minutes apart, inside the 15-minute catchment, which
    holds 100 units of demand. Filling that demand in each block separately
    used to pass.
    """
    acc = pd.DataFrame([[0, 5], [5, 0]], index=[1, 2], columns=[1, 2])
    checker = _capacity_checker(monkeypatch, acc, [50, 50])

    assert checker.get_demand(1, SERVICE) == 110  # 100 plus one rounding-up unit
    assert checker.check_constraints(_placed(6, 5))  # 110 across the shared catchment
    assert not checker.check_constraints(_placed(11, 11))
    assert not checker.check_constraints(_placed(0, 12))


def test_demand_stays_per_block_outside_the_catchment(monkeypatch):
    acc = pd.DataFrame([[0, 999], [999, 0]], index=[1, 2], columns=[1, 2])
    checker = _capacity_checker(monkeypatch, acc, [50, 50])
    assert checker.get_catchment(1, SERVICE) == [1]
    # 60 each: 50 the neighbourhood lacks nearby plus one rounding-up unit.
    assert checker.check_constraints(_placed(6, 6))
    assert not checker.check_constraints(_placed(7, 6))
    assert checker.check_constraints(_placed(6, 5))


# ------------------------------------------------------------ population in the objective


class _StubFacade:
    """Facade stub exposing just what WeightedObjective reads."""

    def __init__(self, provisions, population_ratio):
        self.start_provisions = provisions
        self._provisions = provisions
        self._ratio = population_ratio
        self.num_params = 1

    @property
    def last_population_ratio(self):
        return self._ratio

    def get_provisions(self, _x_last, _x):
        return self._provisions, set(self._provisions)

    def get_X(self, _x):
        return []


def test_objective_rewards_housed_population():
    """Two solutions with equal provision are ranked by the population housed."""
    weights = {"a": 0.6, "b": 0.4}
    provisions = {"a": 1.0, "b": 1.0}
    housed = WeightedObjective(1, _StubFacade(provisions, 1.0), 10, weights, population_weight=1.0)
    emptied = WeightedObjective(1, _StubFacade(provisions, 0.2), 10, weights, population_weight=1.0)
    _, housed_value = housed.evaluate(np.array([1]))
    _, emptied_value = emptied.evaluate(np.array([1]))
    assert housed_value > emptied_value
    assert housed_value == pytest.approx(2.0)
    assert emptied_value == pytest.approx(1.2)


def test_population_is_not_scored_by_default():
    """Scoring stays on provision alone unless a run asks for the housing term."""
    weights = {"a": 0.6, "b": 0.4}
    objective = WeightedObjective(1, _StubFacade({"a": 1.0, "b": 0.5}, 0.0), 10, weights)
    _, value = objective.evaluate(np.array([1]))
    assert objective._population_weight == 0
    assert value == pytest.approx(0.8)


def test_population_weight_none_is_treated_as_off():
    weights = {"a": 0.6, "b": 0.4}
    objective = WeightedObjective(1, _StubFacade({"a": 1.0, "b": 0.5}, 0.0), 10, weights, population_weight=None)
    _, value = objective.evaluate(np.array([1]))
    assert value == pytest.approx(0.8)


# ---------------------------------------------- demand of the new residents


def test_catchment_demand_includes_new_residents(monkeypatch):
    """New residents add to the demand their catchment is measured against.

    Blocks 1 and 2 share a catchment holding 100 units of existing demand.
    Housing people in them adds demand of its own, and services may be built
    for it; before, the block could not serve the people it had just housed.
    """
    acc = pd.DataFrame([[0, 5], [5, 0]], index=[1, 2], columns=[1, 2])
    checker = _capacity_checker(monkeypatch, acc, [50, 50])
    new_demand = {1: {SERVICE: 30}, 2: {SERVICE: 20}}

    assert checker.get_catchment_demand(1, SERVICE) == 110
    assert checker.get_catchment_demand(1, SERVICE, new_demand) == 160  # 110 + 30 + 20

    assert not checker.check_constraints(_placed(9, 8))  # 170 > 110
    assert checker.check_constraints(_placed(9, 7), new_demand)  # 160 <= 160
    assert not checker.check_constraints(_placed(9, 8), new_demand)  # 170 > 160


# ------------------------------------------------- optimizer robustness


def test_fixed_variable_survives_a_pruned_first_trial():
    """A pruned initial trial must not crash the next one.

    With no completed trial there is no solution to hold a gradient-fixed
    variable at, and reading one raised AttributeError.
    """
    import optuna

    from blocksnet.optimization.services.core.optimizer import TPEOptimizer

    class _Objective:
        num_params = 1
        current_func_evals = 0

        def get_penalty(self, _x):
            return 0.0

    class _Constraints:
        def suggest_solution(self, permut, suggest_callback):
            return np.array([suggest_callback(0, 3, 7)])

        def check_constraints(self, _x):
            return False

    optimizer = TPEOptimizer.__new__(TPEOptimizer)
    optimizer._objective = _Objective()
    optimizer._constraints = _Constraints()
    optimizer._study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)
    optimizer._metrics = type("M", (), {"update_metrics": lambda *a, **k: None})()
    optimizer._vars_order = lambda order: order
    optimizer._vars_chooser = lambda order, _cb: ([], list(order))

    trial = optimizer._study.ask()
    trial.set_user_attr("initial", False)
    # trial.number is 0 here, so the fixed branch is not reached; force it by
    # asking for a second trial after the first is discarded.
    optimizer._study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    second = optimizer._study.ask()
    with pytest.raises(optuna.TrialPruned):
        optimizer._optuna_objective(second)


# ------------------------------------------- ground floor and unit choice


def test_built_in_services_are_capped_by_the_ground_floor(monkeypatch):
    """Built-in services fit on the ground floor, not into the whole block.

    The block has room for 100 units by floor area, but only 350 m2 of ground
    floor is open to services, so three of them fit.
    """
    adapter = _built_in_adapter(monkeypatch)
    adapter.update_gfa({BLOCK_ID: 350.0})
    assert sum(var.count for var in adapter(np.array([10_000]))) == 3


def test_ground_floor_budget_comes_from_the_built_footprint():
    from blocksnet.optimization.services.acl.checkers.area_checker import AreaChecker

    blocks = pd.DataFrame(
        {"site_area": [50_000.0], "footprint_area": [10_000.0]},
        index=[BLOCK_ID],
    )
    checker = AreaChecker({BLOCK_ID: LandUse.RESIDENTIAL}, blocks)
    assert checker.ground_floor_areas[BLOCK_ID] == pytest.approx(7_000.0)  # 0.7 of the footprint

    built_in = Variable(
        block_id=BLOCK_ID, service_type=SERVICE, capacity=1, site_area=0, build_floor_area=7_000, count=1
    )
    assert checker.check_constraints([built_in])
    built_in.count = 2
    assert not checker.check_constraints([built_in])


def test_units_are_ordered_by_fewest_facilities(monkeypatch):
    """A service is delivered by as few facilities as possible.

    Ranking by capacity per square metre made the built-in unit win: it offers
    0.1 place per m2 against the standalone unit's 0.03, although the latter is
    one facility for 1100 places instead of five.
    """
    import blocksnet.optimization.services.acl.variable_adapters.area_solution as mod

    units_df = pd.DataFrame(
        [
            {"service_type": SERVICE, "site_area": 0.0, "build_floor_area": 2400.0, "capacity": 250},
            {"service_type": SERVICE, "site_area": 33000.0, "build_floor_area": 6600.0, "capacity": 1100},
            {"service_type": SERVICE, "site_area": 0.0, "build_floor_area": 3900.0, "capacity": 300},
        ]
    )

    class _StubConfig:
        units = units_df

        def __getitem__(self, _key):
            return {SERVICE: None}

    monkeypatch.setattr(mod, "service_types_config", _StubConfig())
    adapter = AreaSolution({BLOCK_ID: LandUse.RESIDENTIAL})
    adapter.add_service_type_vars(SERVICE)
    assert [u.capacity for u in adapter._units_lists[SERVICE]] == [1100, 300, 250]
    # The granted area is bounded through the leading unit, so the one with
    # grounds can actually be granted the 33000 m2 it needs.
    assert adapter.X[0].capacity == pytest.approx(1100 / 33000)


def test_footprint_larger_than_site_area_is_reported():
    """A block covered by more building than it has land means site_area is wrong."""
    from blocksnet.optimization.services.acl.checkers.area_checker import AreaChecker

    blocks = pd.DataFrame({"site_area": [10_000.0], "footprint_area": [10_000.0]}, index=[BLOCK_ID])
    messages = []
    from loguru import logger

    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        AreaChecker({BLOCK_ID: LandUse.RESIDENTIAL}, blocks)
        assert not messages
        blocks.loc[BLOCK_ID, "footprint_area"] = 10_001.0
        AreaChecker({BLOCK_ID: LandUse.RESIDENTIAL}, blocks)
        assert any("footprint_area exceeds site_area" in m for m in messages)
    finally:
        logger.remove(sink)


def test_placement_shares_the_catchment_demand(monkeypatch):
    """Blocks of one catchment cannot each place that catchment's whole demand.

    The suggestion stage already shared the demand, but the adapter expanded
    the granted area against each block's own budget, so a catchment holding
    249 library places received 690.
    """
    import blocksnet.optimization.services.acl.variable_adapters.area_solution as mod

    units_df = pd.DataFrame([{"service_type": SERVICE, "site_area": 0.0, "build_floor_area": 100.0, "capacity": 50}])

    class _StubConfig:
        units = units_df

        def __getitem__(self, _key):
            return {SERVICE: None}

    monkeypatch.setattr(mod, "service_types_config", _StubConfig())

    def placed(dependents):
        adapter = AreaSolution({1: LandUse.RESIDENTIAL, 2: LandUse.RESIDENTIAL})
        adapter.add_service_type_vars(SERVICE)
        adapter.update_bfa({1: 10_000.0, 2: 10_000.0})
        adapter.update_gfa({1: 10_000.0, 2: 10_000.0})
        adapter.update_cap_dependents(dependents)
        adapter.update_cap({1: {SERVICE: 200.0}, 2: {SERVICE: 200.0}})
        return sum(var.total_capacity for var in adapter(np.array([10_000, 10_000])))

    # Both blocks draw on the same 200 places of demand.
    assert placed({SERVICE: {1: [1, 2], 2: [1, 2]}}) == 200
    # Without the shared budget each block takes the full 200 on its own.
    assert placed({}) == 400


def test_facade_initialises_without_a_service_type():
    """Facade.__init__ must wire its checkers in a usable order.

    The other tests build a Facade through __new__, so a NameError-class bug in
    __init__ went unnoticed until a real run hit it.
    """
    from blocksnet.optimization.services.acl.facade import Facade

    blocks = pd.DataFrame(
        {
            "site_area": [50_000.0, 50_000.0],
            "footprint_area": [10_000.0, 10_000.0],
            "population": [0.0, 0.0],
        },
        index=[1, 2],
    )
    accessibility = pd.DataFrame([[0.0, 5.0], [5.0, 0.0]], index=[1, 2], columns=[1, 2])
    blocks_lu = {1: LandUse.RESIDENTIAL, 2: LandUse.RESIDENTIAL}
    facade = Facade(
        var_adapter=AreaSolution(blocks_lu),
        accessibility_matrix=accessibility,
        blocks_df=blocks,
        blocks_lu=blocks_lu,
    )
    assert facade.num_params == 0
    assert facade.max_population > 0


def test_blocks_serve_their_own_residents_by_default(monkeypatch):
    """By default a redevelopment is sized to the people it houses.

    A wide catchment holds far more unmet demand than the redevelopment
    generates, and taking it on is what let one block supply a whole district.
    """
    acc = pd.DataFrame([[0, 5], [5, 0]], index=[1, 2], columns=[1, 2])
    checker = _capacity_checker(monkeypatch, acc, [50, 50], neighbourhood_share=0.0)
    new_demand = {1: {SERVICE: 30}, 2: {SERVICE: 20}}

    # The neighbourhood's own 100 unmet places are not the project's to serve.
    assert checker.get_catchment_demand(1, SERVICE) == 0
    assert checker.get_catchment_demand(1, SERVICE, new_demand) == 50
    assert checker.check_constraints(_placed(3, 2), new_demand)  # 50 of 50
    assert not checker.check_constraints(_placed(3, 3), new_demand)  # 60 of 50


def test_neighbourhood_share_is_a_fraction(monkeypatch):
    acc = pd.DataFrame([[0, 5], [5, 0]], index=[1, 2], columns=[1, 2])
    half = _capacity_checker(monkeypatch, acc, [50, 50], neighbourhood_share=0.5)
    assert half.get_catchment_demand(1, SERVICE) == 55  # half of 110
    with pytest.raises(ValueError, match="between 0 and 1"):
        CapacityChecker([1], acc, neighbourhood_share=1.5)


def test_priority_orders_who_claims_budget_first():
    """`priority` must actually order the variables; it used to be ignored.

    Block budgets are handed out as the variables are walked, so the order
    decides who gets the room. The argument was stored and never read.
    """
    from blocksnet.optimization.services.core.constraints import WeightedConstraints

    class _StubFacade:
        def get_upper_bound_var(self, _n):
            return 1

        def get_var_block_id(self, var_num):
            return 1

        def get_service_name(self, var_num):
            return ["dog_park", "school", "bar", "kindergarten"][var_num]

        def get_var_weights(self, _n):
            return np.array([1.0, 1.0, 1.0])

        def get_block_services(self, _b):
            return {"dog_park", "school", "bar", "kindergarten"}

        def get_capacity_dependent_blocks(self, _n):
            return [1]

        def get_min_useful_grant(self, _n):
            return 0.0

        def get_var_bfa_share(self, _n):
            return 0.0

        residents_per_bfa = 0.09333

        def get_service_demand_norm(self, _s):
            return 1.0

    priority = {"school": 0.9, "kindergarten": 0.8, "bar": 0.2, "dog_park": 0.1}
    constraints = WeightedConstraints(facade=_StubFacade(), num_params=4, priority=priority)

    assert constraints._in_priority_order([0, 1, 2, 3]) == [1, 3, 2, 0]
    # Ties keep the order they came in, so variable ordering still explores.
    same = {"school": 0.5, "kindergarten": 0.5, "bar": 0.5, "dog_park": 0.5}
    flat = WeightedConstraints(facade=_StubFacade(), num_params=4, priority=same)
    assert flat._in_priority_order([3, 1, 0, 2]) == [3, 1, 0, 2]


def test_eval_budget_scales_with_the_catalogue():
    """The budget is spent per service type, so it must grow with their number.

    A fixed 200 bought ~28 trials over nine service types and 8 over 33.
    """
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "run_17blocks_joint", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_17blocks_joint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.resolve_max_evals(None, 120, 33) == 120 * 33
    assert module.resolve_max_evals(None, 120, 7) == 120 * 7
    assert module.resolve_max_evals(200, 120, 33) == 200  # explicit wins
    assert module.resolve_max_evals(None, 120, 0) == 120  # never zero


def test_reported_solution_does_not_depend_on_converter_state(monkeypatch):
    """The same solution must render the same, whatever ran before it.

    The converter accumulates the previous trial's solution on every other
    call. Reporting through it made one run describe `best_x` as 169 services
    and another, for a byte-identical `best_x`, as 102.
    """
    import blocksnet.optimization.services.acl.variable_adapters.area_solution as mod
    from blocksnet.optimization.services.acl.facade import Facade

    units_df = pd.DataFrame([{"service_type": SERVICE, "site_area": 0.0, "build_floor_area": 100.0, "capacity": 50}])

    class _StubConfig:
        units = units_df

        def __getitem__(self, _key):
            return {SERVICE: None}

    monkeypatch.setattr(mod, "service_types_config", _StubConfig())

    blocks_lu = {BLOCK_ID: LandUse.RESIDENTIAL}
    adapter = AreaSolution(blocks_lu)
    adapter.add_service_type_vars(SERVICE)
    adapter.update_bfa({BLOCK_ID: 10_000.0})
    adapter.update_gfa({BLOCK_ID: 10_000.0})
    adapter.update_cap({BLOCK_ID: {SERVICE: 10_000.0}})

    facade = Facade.__new__(Facade)
    facade._converter = adapter
    facade.num_params = 1

    solution = {"x_0": 500}
    fresh = facade.solution_to_services_df(solution)

    # Leave the converter carrying a previous solution, as a finished run does.
    adapter(np.array([500]))
    adapter.update_x_prev()
    assert adapter._X_prev is not None

    again = facade.solution_to_services_df(solution)
    pd.testing.assert_frame_equal(fresh.reset_index(drop=True), again.reset_index(drop=True))
    assert int(again["count"].sum()) == 5


# ------------------------------------------------------- variable choosing


class _ChooserFacade:
    """Facade stub: four variables, one per service type, equal area."""

    SERVICES = ["school", "bus_stop", "kindergarten", "substation"]

    def get_service_name(self, var_num):
        return self.SERVICES[var_num]

    def get_var_weights(self, var_num):
        # school takes a big plot, the others a small one
        return np.array([1000.0 if var_num == 0 else 10.0, 0.0, 0.0])


def _chooser(num_top=2, weights=None):
    from blocksnet.optimization.services.core.variable_choosing import GradientChooser

    return GradientChooser(_ChooserFacade(), num_params=4, num_top=num_top, weights=weights)


def test_gradient_ranks_by_the_provision_each_service_gained():
    """The key must use each service's own provision, not the objective total.

    The objective's change is one number shared by every service, so dividing
    it by each service's area ranked them by who spent least, and a school on a
    big plot could never outrank a bus stop on a small one.
    """
    chooser = _chooser(num_top=2)
    # Both moved. Per square metre the school bought more: 0.80/1000 against
    # 0.001/10. The old key could not see this at all — it divided one shared
    # number by each service's area, so the 10 m2 service always won.
    previous = [1.0, np.array([1, 1, 0, 0]), {"school": 0.100, "bus_stop": 0.500}]
    last = [1.4, np.array([2, 2, 0, 0]), {"school": 0.900, "bus_stop": 0.501}]
    opt, fixed = chooser._choose([0, 1, 2, 3], lambda: (previous, last))
    assert "school" in {_ChooserFacade().get_service_name(v) for v in opt}
    # Nothing is dropped from both lists any more.
    assert sorted([*opt, *fixed]) == [0, 1, 2, 3]


def test_untried_services_get_a_share_of_the_slots():
    """A service can only earn a gradient once it has been moved."""
    chooser = _chooser(num_top=4)
    previous = [1.0, np.array([1, 1, 0, 0]), {"school": 0.1, "bus_stop": 0.5}]
    last = [1.4, np.array([2, 2, 0, 0]), {"school": 0.9, "bus_stop": 0.51}]
    opt, _ = chooser._choose([0, 1, 2, 3], lambda: (previous, last))
    chosen = {_ChooserFacade().get_service_name(v) for v in opt}
    assert chosen & {"kindergarten", "substation"}, "never-moved services must get a turn"


def test_weights_scale_the_gradient():
    """A service worth more to the objective outranks an equally efficient one."""
    # Equal gain per square metre: 0.10/1000 for the school, 0.001/10 for the stop.
    previous = [1.0, np.array([1, 1, 0, 0]), {"school": 0.0, "bus_stop": 0.0}]
    last = [1.4, np.array([2, 2, 0, 0]), {"school": 0.100, "bus_stop": 0.001}]

    unweighted = _chooser(num_top=1)._choose([0, 1], lambda: (previous, last))[0]
    assert len(unweighted) == 1  # a tie, broken arbitrarily

    weighted = _chooser(num_top=1, weights={"school": 10.0, "bus_stop": 0.1})
    opt, _ = weighted._choose([0, 1], lambda: (previous, last))
    assert {_ChooserFacade().get_service_name(v) for v in opt} == {"school"}


def test_default_chooser_is_constructible():
    """The default used to be `WeightChooser()`, which raises TypeError."""
    from blocksnet.optimization.services.core.variable_choosing import SimpleChooser, WeightChooser

    opt, fixed = SimpleChooser()._choose(np.array([0, 1, 2]), lambda: None)
    assert list(opt) == [0, 1, 2] and len(fixed) == 0

    # WeightChooser's own sort was passed positionally and raised as well.
    chooser = WeightChooser(_ChooserFacade(), weights={s: 1.0 for s in _ChooserFacade.SERVICES}, num_top=2)
    chooser._choose([0, 1, 2, 3], lambda: None)


def test_plan_comes_from_recorded_counts_not_from_rereading(monkeypatch):
    """A solution vector does not describe a plan on its own.

    The converter reads a vector against the previous trial's solution, so the
    same vector maps to different plans depending on what ran before — and the
    re-read plan may even break the constraints, which is what the run's own
    check caught. The counts a trial recorded are the plan it was scored on.
    """
    import blocksnet.optimization.services.acl.variable_adapters.area_solution as mod
    from blocksnet.optimization.services.acl.facade import Facade

    units_df = pd.DataFrame([{"service_type": SERVICE, "site_area": 0.0, "build_floor_area": 100.0, "capacity": 50}])

    class _StubConfig:
        units = units_df

        def __getitem__(self, _key):
            return {SERVICE: None}

    monkeypatch.setattr(mod, "service_types_config", _StubConfig())

    adapter = AreaSolution({BLOCK_ID: LandUse.RESIDENTIAL})
    adapter.add_service_type_vars(SERVICE)
    facade = Facade.__new__(Facade)
    facade._converter = adapter
    facade.num_params = 1

    df = facade.services_df_from_counts([4])
    assert int(df["count"].sum()) == 4
    assert facade.variables_from_counts([4])[0].count == 4

    with pytest.raises(Exception):
        facade.services_df_from_counts([1, 2, 3])


# ------------------------------------------------------------ dead zone


class _GrantFacade:
    """One variable, one block, one service whose smallest facility is 430 m²."""

    def get_upper_bound_var(self, _n):
        return 5000

    def get_var_block_id(self, _n):
        return 1

    def get_service_name(self, _n):
        return SERVICE

    def get_var_weights(self, _n):
        return np.array([1.0, 1.0, 0.0])

    def get_block_services(self, _b):
        return {SERVICE}

    def get_capacity_dependent_blocks(self, _n):
        return [1]

    def get_min_useful_grant(self, _n):
        return 430.0

    def get_var_bfa_share(self, _n):
        return 0.0

    residents_per_bfa = 0.09333

    def get_service_demand_norm(self, _s):
        return 61.0

    def get_limits_var(self, _n):
        return np.array([10_000.0, 10_000.0, 10_000.0])

    def get_lower_bound_var_val(self, _n):
        return 430  # one facility

    def get_var_land_use(self, _n):
        return LandUse.RESIDENTIAL


def _grant_constraints():
    from blocksnet.optimization.services.core.constraints import WeightedConstraints

    return WeightedConstraints(facade=_GrantFacade(), num_params=1, priority={SERVICE: 1.0})


def test_grant_too_small_to_build_is_rounded_up():
    """A grant below the smallest facility buys nothing but costs budget.

    The kindergarten needs 430 m² for one place of any size; a grant of 200
    produced no service, spent the block's area and demand all the same, and
    left the objective unmoved, so the optimizer could not learn from it.
    """
    constraints = _grant_constraints()
    limits = {SERVICE: 10_000.0}
    assert constraints._snap_out_of_dead_zone(0, 200, 0, [10_000.0, 10_000.0], {1: limits}, [1]) == 430


def test_grant_too_small_falls_to_nothing_when_it_cannot_be_raised():
    """If a whole facility does not fit, the grant is released, not wasted."""
    constraints = _grant_constraints()
    limits = {SERVICE: 10_000.0}
    assert constraints._snap_out_of_dead_zone(0, 200, 0, [300.0, 300.0], {1: limits}, [1]) == 0


def test_grants_that_already_build_are_left_alone():
    constraints = _grant_constraints()
    limits = {SERVICE: 10_000.0}
    assert constraints._snap_out_of_dead_zone(0, 900, 0, [10_000.0, 10_000.0], {1: limits}, [1]) == 900
    assert constraints._snap_out_of_dead_zone(0, 0, 0, [10_000.0, 10_000.0], {1: limits}, [1]) == 0
    # Measured against what the previous solution already granted.
    assert constraints._snap_out_of_dead_zone(0, 1000, 900, [10_000.0, 10_000.0], {1: limits}, [1]) == 1330


def test_demand_is_counted_inside_the_service_radius(monkeypatch):
    """Everyone the service reaches counts: the city and the blocks nearby.

    Both parts are taken inside the radius of the service in question, so a
    block is not credited with a facility an hour's walk away.
    """
    together = pd.DataFrame([[0, 5], [5, 0]], index=[1, 2], columns=[1, 2])
    checker = _capacity_checker(monkeypatch, together, [50, 50], neighbourhood_share=1.0)
    new_demand = {1: {SERVICE: 30}, 2: {SERVICE: 20}}
    # 110 the neighbourhood still lacks in reach, plus 50 from both blocks.
    assert checker.get_catchment_demand(1, SERVICE, new_demand) == 160

    apart = pd.DataFrame([[0, 999], [999, 0]], index=[1, 2], columns=[1, 2])
    far = _capacity_checker(monkeypatch, apart, [50, 50], neighbourhood_share=1.0)
    # Out of reach of each other, a block answers for its own residents only:
    # 60 the neighbourhood lacks nearby plus this block's own 30.
    assert far.get_catchment_demand(1, SERVICE, new_demand) == 90




def test_basic_services_are_filled_before_the_rest():
    """Tiers must decide the fill order; the objective weight cannot.

    A tier's share multiplies a weight that runs from 0.1 to 0.8 inside the
    tier, so the bands overlap: a hospital scored 0.5 × 0.2857 = 0.143 against
    a kindergarten's 0.2 × 0.5714 = 0.114 and claimed a block's budget first.
    """
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "run_17blocks_joint", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_17blocks_joint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    types = ["kindergarten", "school", "hospital", "religion", "bar", "landfill"]
    weights = {st: module.get_service_weight(st) for st in types}
    basic = min(weights["kindergarten"], weights["school"])
    assert all(weights[st] < basic for st in ["hospital", "religion", "bar", "landfill"])
    # The bands never meet, whatever the weights inside a tier.
    assert weights["hospital"] > weights["bar"] > weights["landfill"]



from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import pyomo.environ as pyo


# =========================
# Data container
# =========================
@dataclass
class MRPData:
    products: List[str]
    periods: List[str]
    locations: List[str]

    # Independent demand by location
    demand: Dict[str, Dict[str, Dict[str, float]]]
    initial_inventory: Dict[str, Dict[str, float]]

    # MAKE capacity (production)
    capacity: Dict[str, Dict[str, Dict[str, float]]]

    # BUY capacity (optional, can be empty -> unlimited)
    cap_buy: Dict[str, Dict[str, Dict[str, float]]]

    # BOM by location: parent -> location -> component -> qty
    bom: Dict[str, Dict[str, Dict[str, float]]]

    # Procurement type by location: "P"=make, "F"=buy, "X"=both
    proc_type: Dict[str, Dict[str, str]]

    # Lead times (release -> receipt)
    lt_make: Dict[str, Dict[str, int]]
    lt_buy: Dict[str, Dict[str, int]]

    # Lot sizing
    min_lot_make: Dict[str, Dict[str, float]]
    multiple_lot_make: Dict[str, Dict[str, int]]
    min_lot_buy: Dict[str, Dict[str, float]]
    multiple_lot_buy: Dict[str, Dict[str, int]]

    # Transfers (by lane)
    ship_allowed: Dict[str, Dict[str, Dict[str, bool]]]
    ship_priority: Dict[str, Dict[str, Dict[str, int]]]
    ship_cap: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]
    lt_ship: Dict[str, Dict[str, Dict[str, int]]]

    allow_backlog: bool = True


# =========================
# Model builder
# =========================
def build_mrp_model(data: MRPData) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel("MRP_MakeBuy")

    # -----------------
    # Sets
    # -----------------
    m.P = pyo.Set(initialize=data.products)
    m.T = pyo.Set(initialize=data.periods, ordered=True)
    m.L = pyo.Set(initialize=data.locations)

    m.T_list = list(data.periods)
    m.T_index = {t: i for i, t in enumerate(m.T_list)}

    # -----------------
    # Parameters
    # -----------------
    m.d_ind = pyo.Param(
        m.P, m.L, m.T,
        initialize=lambda _, p, l, t: float(data.demand.get(p, {}).get(l, {}).get(t, 0.0)),
        default=0.0
    )

    m.cap_make = pyo.Param(
        m.P, m.L, m.T,
        initialize=lambda _, p, l, t: float(data.capacity.get(p, {}).get(l, {}).get(t, 0.0)),
        default=0.0
    )

    def cap_buy_init(_, p, l, t):
        if p in data.cap_buy and l in data.cap_buy[p] and t in data.cap_buy[p][l]:
            return float(data.cap_buy[p][l][t])
        return 1e12  # unlimited if not provided

    m.cap_buy = pyo.Param(m.P, m.L, m.T, initialize=cap_buy_init)

    # store as python dict; used in constraints via mm.I0.get(...)
    m.I0 = data.initial_inventory

    m.bom = pyo.Param(
        m.P, m.L, m.P,
        initialize=lambda _, parent, l, comp: float(data.bom.get(parent, {}).get(l, {}).get(comp, 0.0)),
        default=0.0
    )

    def make_allowed_init(_, p, l):
        return 1 if data.proc_type.get(p, {}).get(l, "X") in ("P", "X") else 0

    def buy_allowed_init(_, p, l):
        return 1 if data.proc_type.get(p, {}).get(l, "X") in ("F", "X") else 0

    m.make_allowed = pyo.Param(m.P, m.L, initialize=make_allowed_init)
    m.buy_allowed = pyo.Param(m.P, m.L, initialize=buy_allowed_init)

    m.lt_make = pyo.Param(
        m.P, m.L, initialize=lambda _, p, l: int(data.lt_make.get(p, {}).get(l, 0)), default=0
    )
    m.lt_buy = pyo.Param(
        m.P, m.L, initialize=lambda _, p, l: int(data.lt_buy.get(p, {}).get(l, 0)), default=0
    )

    m.min_lot_make = pyo.Param(
        m.P, m.L, initialize=lambda _, p, l: float(data.min_lot_make.get(p, {}).get(l, 0.0)), default=0.0
    )
    m.mult_lot_make = pyo.Param(
        m.P, m.L, initialize=lambda _, p, l: int(data.multiple_lot_make.get(p, {}).get(l, 1)), default=1
    )

    m.min_lot_buy = pyo.Param(
        m.P, m.L, initialize=lambda _, p, l: float(data.min_lot_buy.get(p, {}).get(l, 0.0)), default=0.0
    )
    m.mult_lot_buy = pyo.Param(
        m.P, m.L, initialize=lambda _, p, l: int(data.multiple_lot_buy.get(p, {}).get(l, 1)), default=1
    )

    # -----------------
    # Decision variables (releases)
    # -----------------
    m.k_make = pyo.Var(m.P, m.L, m.T, domain=pyo.NonNegativeIntegers)
    m.k_buy = pyo.Var(m.P, m.L, m.T, domain=pyo.NonNegativeIntegers)

    m.r_make = pyo.Expression(
        m.P, m.L, m.T, rule=lambda mm, p, l, t: mm.k_make[p, l, t] * mm.mult_lot_make[p, l]
    )
    m.r_buy = pyo.Expression(
        m.P, m.L, m.T, rule=lambda mm, p, l, t: mm.k_buy[p, l, t] * mm.mult_lot_buy[p, l]
    )

    m.y_make = pyo.Var(m.P, m.L, m.T, domain=pyo.Binary)
    m.y_buy = pyo.Var(m.P, m.L, m.T, domain=pyo.Binary)

    # -----------------
    # Receipts implied by releases + lead times
    # -----------------
    def receipt_expr(mm, p, l, t):
        idx = mm.T_index[t]
        val = 0.0

        lm = int(pyo.value(mm.lt_make[p, l]))
        lb = int(pyo.value(mm.lt_buy[p, l]))

        if idx - lm >= 0:
            val += mm.r_make[p, l, mm.T_list[idx - lm]]
        if idx - lb >= 0:
            val += mm.r_buy[p, l, mm.T_list[idx - lb]]

        return val

    m.x = pyo.Expression(m.P, m.L, m.T, rule=receipt_expr)

    # -----------------
    # Inventory / backlog
    # -----------------
    m.I = pyo.Var(m.P, m.L, m.T, domain=pyo.NonNegativeReals)
    m.B = pyo.Var(m.P, m.L, m.T, domain=pyo.NonNegativeReals)

    if not data.allow_backlog:
        for p in data.products:
            for l in data.locations:
                for t in data.periods:
                    m.B[p, l, t].fix(0.0)

    # -----------------
    # Constraints
    # -----------------
    BIG_M = 1e12

    # Make / Buy allowed
    m.MakeAllowed = pyo.Constraint(
        m.P, m.L, m.T, rule=lambda mm, p, l, t: mm.k_make[p, l, t] <= BIG_M * mm.make_allowed[p, l]
    )
    m.BuyAllowed = pyo.Constraint(
        m.P, m.L, m.T, rule=lambda mm, p, l, t: mm.k_buy[p, l, t] <= BIG_M * mm.buy_allowed[p, l]
    )

    # Capacity
    m.MakeCap = pyo.Constraint(
        m.P, m.L, m.T, rule=lambda mm, p, l, t: mm.r_make[p, l, t] <= mm.cap_make[p, l, t]
    )
    m.BuyCap = pyo.Constraint(
        m.P, m.L, m.T, rule=lambda mm, p, l, t: mm.r_buy[p, l, t] <= mm.cap_buy[p, l, t]
    )

    # Lot sizing
    m.MinLotMakeLB = pyo.Constraint(
        m.P, m.L, m.T, rule=lambda mm, p, l, t: mm.r_make[p, l, t] >= mm.min_lot_make[p, l] * mm.y_make[p, l, t]
    )
    m.MinLotMakeUB = pyo.Constraint(
        m.P, m.L, m.T, rule=lambda mm, p, l, t: mm.r_make[p, l, t] <= BIG_M * mm.y_make[p, l, t]
    )

    m.MinLotBuyLB = pyo.Constraint(
        m.P, m.L, m.T, rule=lambda mm, p, l, t: mm.r_buy[p, l, t] >= mm.min_lot_buy[p, l] * mm.y_buy[p, l, t]
    )
    m.MinLotBuyUB = pyo.Constraint(
        m.P, m.L, m.T, rule=lambda mm, p, l, t: mm.r_buy[p, l, t] <= BIG_M * mm.y_buy[p, l, t]
    )

    # Transfers (shipments)
    m.ship = pyo.Var(m.P, m.L, m.L, m.T, domain=pyo.NonNegativeReals)

    def ship_allowed_init(_, p, lf, lt):
        return 1 if data.ship_allowed.get(p, {}).get(lf, {}).get(lt, False) else 0

    m.ship_allowed = pyo.Param(m.P, m.L, m.L, initialize=ship_allowed_init, default=0)

    def ship_prio_init(_, p, lf, lt):
        return int(data.ship_priority.get(p, {}).get(lf, {}).get(lt, 0))

    m.ship_priority = pyo.Param(m.P, m.L, m.L, initialize=ship_prio_init, default=0)

    def ship_cap_init(_, p, lf, lt, t):
        if p in data.ship_cap and lf in data.ship_cap[p] and lt in data.ship_cap[p][lf] and t in data.ship_cap[p][lf][lt]:
            return float(data.ship_cap[p][lf][lt][t])
        return 1e12

    m.ship_cap = pyo.Param(m.P, m.L, m.L, m.T, initialize=ship_cap_init)

    def ship_lt_init(_, p, lf, lt):
        return int(data.lt_ship.get(p, {}).get(lf, {}).get(lt, 0))

    m.lt_ship = pyo.Param(m.P, m.L, m.L, initialize=ship_lt_init, default=0)

    # Disallow non-allowed lanes
    m.ShipAllowed = pyo.Constraint(
        m.P, m.L, m.L, m.T, rule=lambda mm, p, lf, lt, t: mm.ship[p, lf, lt, t] <= BIG_M * mm.ship_allowed[p, lf, lt]
    )

    # Capacity per lane/time
    m.ShipCap = pyo.Constraint(
        m.P, m.L, m.L, m.T, rule=lambda mm, p, lf, lt, t: mm.ship[p, lf, lt, t] <= mm.ship_cap[p, lf, lt, t]
    )

    def ship_receipt_expr(mm, p, lf, lt, t):
        idx = mm.T_index[t]
        lts = int(pyo.value(mm.lt_ship[p, lf, lt]))
        if idx - lts < 0:
            return 0.0
        return mm.ship[p, lf, lt, mm.T_list[idx - lts]]

    m.ship_receipt = pyo.Expression(m.P, m.L, m.L, m.T, rule=ship_receipt_expr)

    # Inventory balance (with initial inventory as starting stock)
    def dep_demand_expr(mm, p, l, t):
        return sum(mm.bom[parent, l, p] * mm.r_make[parent, l, t] for parent in mm.P)

    m.d_dep = pyo.Expression(m.P, m.L, m.T, rule=dep_demand_expr)

    def balance_rule(mm, p, l, t):
        idx = mm.T_index[t]

        if idx == 0:
            prev_net = float(mm.I0.get(p, {}).get(l, 0.0))  # initial backlog assumed 0
        else:
            pt = mm.T_list[idx - 1]
            prev_net = mm.I[p, l, pt] - mm.B[p, l, pt]

        inbound = sum(mm.ship_receipt[p, lf, l, t] for lf in mm.L)
        outbound = sum(mm.ship[p, l, lt, t] for lt in mm.L)

        return (mm.I[p, l, t] - mm.B[p, l, t]) == (
            prev_net
            + mm.x[p, l, t]
            + inbound
            - mm.d_ind[p, l, t]
            - mm.d_dep[p, l, t]
            - outbound
        )

    m.Balance = pyo.Constraint(m.P, m.L, m.T, rule=balance_rule)

    # Anchor: make initial inventory "visible" in SOH (inventory variable) in first period.
    # This forces the model to carry initial stock forward unless it is consumed.
    first_period = m.T_list[0]

    def init_inv_rule(mm, p, l):
        # Net available at end of first period must at least reflect initial stock
        # plus receipts minus demand/consumption captured by Balance.
        # This constraint mainly prevents "hiding" initial stock outside I[p,first_period].
        return (mm.I[p, l, first_period] - mm.B[p, l, first_period]) >= float(mm.I0.get(p, {}).get(l, 0.0)) - 1e-9

    m.InitInventory = pyo.Constraint(m.P, m.L, rule=init_inv_rule)

    # Placeholder objective
    m.Obj = pyo.Objective(expr=0.0, sense=pyo.minimize)

    return m

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

    demand: Dict[str, Dict[str, float]]
    initial_inventory: Dict[str, float]

    # MAKE capacity (production)
    capacity: Dict[str, Dict[str, float]]

    # BUY capacity (optional, can be empty -> unlimited)
    cap_buy: Dict[str, Dict[str, float]]

    # BOM: parent -> component -> qty
    bom: Dict[str, Dict[str, float]]

    # Make / Buy permissions
    make_allowed: Dict[str, bool]
    buy_allowed: Dict[str, bool]

    # Lead times (release -> receipt)
    lt_make: Dict[str, int]
    lt_buy: Dict[str, int]

    # Lot sizing
    min_lot_make: Dict[str, float]
    multiple_lot_make: Dict[str, int]
    min_lot_buy: Dict[str, float]
    multiple_lot_buy: Dict[str, int]

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

    m.T_list = list(data.periods)
    m.T_index = {t: i for i, t in enumerate(m.T_list)}

    # -----------------
    # Parameters
    # -----------------
    m.d = pyo.Param(
        m.P, m.T,
        initialize=lambda _, p, t: float(data.demand.get(p, {}).get(t, 0.0)),
        default=0.0
    )

    m.cap_make = pyo.Param(
        m.P, m.T,
        initialize=lambda _, p, t: float(data.capacity.get(p, {}).get(t, 0.0)),
        default=0.0
    )

    def cap_buy_init(_, p, t):
        if p in data.cap_buy and t in data.cap_buy[p]:
            return float(data.cap_buy[p][t])
        return 1e12  # unlimited if not provided

    m.cap_buy = pyo.Param(m.P, m.T, initialize=cap_buy_init)

    # store as python dict; used in constraints via mm.I0.get(...)
    m.I0 = data.initial_inventory

    m.bom = pyo.Param(
        m.P, m.P,
        initialize=lambda _, parent, comp: float(data.bom.get(parent, {}).get(comp, 0.0)),
        default=0.0
    )

    m.make_allowed = pyo.Param(
        m.P, initialize=lambda _, p: 1 if data.make_allowed.get(p, False) else 0
    )
    m.buy_allowed = pyo.Param(
        m.P, initialize=lambda _, p: 1 if data.buy_allowed.get(p, False) else 0
    )

    m.lt_make = pyo.Param(
        m.P, initialize=lambda _, p: int(data.lt_make.get(p, 0)), default=0
    )
    m.lt_buy = pyo.Param(
        m.P, initialize=lambda _, p: int(data.lt_buy.get(p, 0)), default=0
    )

    m.min_lot_make = pyo.Param(
        m.P, initialize=lambda _, p: float(data.min_lot_make.get(p, 0.0)), default=0.0
    )
    m.mult_lot_make = pyo.Param(
        m.P, initialize=lambda _, p: int(data.multiple_lot_make.get(p, 1)), default=1
    )

    m.min_lot_buy = pyo.Param(
        m.P, initialize=lambda _, p: float(data.min_lot_buy.get(p, 0.0)), default=0.0
    )
    m.mult_lot_buy = pyo.Param(
        m.P, initialize=lambda _, p: int(data.multiple_lot_buy.get(p, 1)), default=1
    )

    # -----------------
    # Decision variables (releases)
    # -----------------
    m.k_make = pyo.Var(m.P, m.T, domain=pyo.NonNegativeIntegers)
    m.k_buy = pyo.Var(m.P, m.T, domain=pyo.NonNegativeIntegers)

    m.r_make = pyo.Expression(
        m.P, m.T, rule=lambda mm, p, t: mm.k_make[p, t] * mm.mult_lot_make[p]
    )
    m.r_buy = pyo.Expression(
        m.P, m.T, rule=lambda mm, p, t: mm.k_buy[p, t] * mm.mult_lot_buy[p]
    )

    m.y_make = pyo.Var(m.P, m.T, domain=pyo.Binary)
    m.y_buy = pyo.Var(m.P, m.T, domain=pyo.Binary)

    # -----------------
    # Receipts implied by releases + lead times
    # -----------------
    def receipt_expr(mm, p, t):
        idx = mm.T_index[t]
        val = 0.0

        lm = int(pyo.value(mm.lt_make[p]))
        lb = int(pyo.value(mm.lt_buy[p]))

        if idx - lm >= 0:
            val += mm.r_make[p, mm.T_list[idx - lm]]
        if idx - lb >= 0:
            val += mm.r_buy[p, mm.T_list[idx - lb]]

        return val

    m.x = pyo.Expression(m.P, m.T, rule=receipt_expr)

    # -----------------
    # Inventory / backlog
    # -----------------
    m.I = pyo.Var(m.P, m.T, domain=pyo.NonNegativeReals)
    m.B = pyo.Var(m.P, m.T, domain=pyo.NonNegativeReals)

    if not data.allow_backlog:
        for p in data.products:
            for t in data.periods:
                m.B[p, t].fix(0.0)

    # -----------------
    # Constraints
    # -----------------
    BIG_M = 1e12

    # Make / Buy allowed
    m.MakeAllowed = pyo.Constraint(
        m.P, m.T, rule=lambda mm, p, t: mm.k_make[p, t] <= BIG_M * mm.make_allowed[p]
    )
    m.BuyAllowed = pyo.Constraint(
        m.P, m.T, rule=lambda mm, p, t: mm.k_buy[p, t] <= BIG_M * mm.buy_allowed[p]
    )

    # Capacity
    m.MakeCap = pyo.Constraint(
        m.P, m.T, rule=lambda mm, p, t: mm.r_make[p, t] <= mm.cap_make[p, t]
    )
    m.BuyCap = pyo.Constraint(
        m.P, m.T, rule=lambda mm, p, t: mm.r_buy[p, t] <= mm.cap_buy[p, t]
    )

    # Lot sizing
    m.MinLotMakeLB = pyo.Constraint(
        m.P, m.T, rule=lambda mm, p, t: mm.r_make[p, t] >= mm.min_lot_make[p] * mm.y_make[p, t]
    )
    m.MinLotMakeUB = pyo.Constraint(
        m.P, m.T, rule=lambda mm, p, t: mm.r_make[p, t] <= BIG_M * mm.y_make[p, t]
    )

    m.MinLotBuyLB = pyo.Constraint(
        m.P, m.T, rule=lambda mm, p, t: mm.r_buy[p, t] >= mm.min_lot_buy[p] * mm.y_buy[p, t]
    )
    m.MinLotBuyUB = pyo.Constraint(
        m.P, m.T, rule=lambda mm, p, t: mm.r_buy[p, t] <= BIG_M * mm.y_buy[p, t]
    )

    # Inventory balance (with initial inventory as starting stock)
    def balance_rule(mm, p, t):
        idx = mm.T_index[t]

        if idx == 0:
            prev_net = float(mm.I0.get(p, 0.0))  # initial backlog assumed 0
        else:
            pt = mm.T_list[idx - 1]
            prev_net = mm.I[p, pt] - mm.B[p, pt]

        consumption = sum(mm.bom[parent, p] * mm.r_make[parent, t] for parent in mm.P)

        return (mm.I[p, t] - mm.B[p, t]) == prev_net + mm.x[p, t] - mm.d[p, t] - consumption

    m.Balance = pyo.Constraint(m.P, m.T, rule=balance_rule)

    # Anchor: make initial inventory "visible" in SOH (inventory variable) in first period.
    # This forces the model to carry initial stock forward unless it is consumed.
    first_period = m.T_list[0]

    def init_inv_rule(mm, p):
        # Net available at end of first period must at least reflect initial stock
        # plus receipts minus demand/consumption captured by Balance.
        # This constraint mainly prevents "hiding" initial stock outside I[p,first_period].
        return (mm.I[p, first_period] - mm.B[p, first_period]) >= float(mm.I0.get(p, 0.0)) - 1e-9

    m.InitInventory = pyo.Constraint(m.P, rule=init_inv_rule)

    # Placeholder objective
    m.Obj = pyo.Objective(expr=0.0, sense=pyo.minimize)

    return m

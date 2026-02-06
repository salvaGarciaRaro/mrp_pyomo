import json
from typing import Dict, Any

import pyomo.environ as pyo

from mrp_model import MRPData, build_mrp_model


def load_data(path: str) -> MRPData:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)

    return MRPData(
        products=raw["products"],
        periods=raw["periods"],
        demand=raw.get("demand", {}),
        initial_inventory=raw["initial_inventory"],
        capacity=raw["capacity"],
        cap_buy=raw.get("cap_buy", {}),
        bom=raw.get("bom", {}),

        make_allowed=raw.get("make_allowed", {}),
        buy_allowed=raw.get("buy_allowed", {}),

        lt_make=raw.get("lt_make", {}),
        lt_buy=raw.get("lt_buy", {}),

        min_lot_make=raw.get("min_lot_make", {}),
        multiple_lot_make=raw.get("multiple_lot_make", {}),
        min_lot_buy=raw.get("min_lot_buy", {}),
        multiple_lot_buy=raw.get("multiple_lot_buy", {}),

        allow_backlog=bool(raw.get("allow_backlog", True)),
    )


def solve_three_phase(m: pyo.ConcreteModel, solver_name: str = "highs"):
    solver = pyo.SolverFactory(solver_name)
    if solver is None or not solver.available():
        raise RuntimeError("HiGHS solver not available (pip install highspy)")

    # Phase A: minimize backlog
    if hasattr(m, "Obj"):
        m.del_component(m.Obj)

    m.Obj = pyo.Objective(
        expr=sum(m.B[p, t] for p in m.P for t in m.T),
        sense=pyo.minimize
    )
    solver.solve(m)

    best_backlog = pyo.value(sum(m.B[p, t] for p in m.P for t in m.T))

    # Phase B: minimize inventory
    m.KeepBacklog = pyo.Constraint(
        expr=sum(m.B[p, t] for p in m.P for t in m.T) <= best_backlog + 1e-6
    )

    m.del_component(m.Obj)
    m.Obj = pyo.Objective(
        expr=sum(m.I[p, t] for p in m.P for t in m.T),
        sense=pyo.minimize
    )
    solver.solve(m)

    best_inventory = pyo.value(sum(m.I[p, t] for p in m.P for t in m.T))

    # Phase C: minimize BUY (MAKE preferred)
    m.KeepInventory = pyo.Constraint(
        expr=sum(m.I[p, t] for p in m.P for t in m.T) <= best_inventory + 1e-6
    )

    m.del_component(m.Obj)
    m.Obj = pyo.Objective(
        expr=sum(m.r_buy[p, t] for p in m.P for t in m.T),
        sense=pyo.minimize
    )
    solver.solve(m)

    return {
        "backlog": best_backlog,
        "inventory": best_inventory,
        "buy_volume": pyo.value(sum(m.r_buy[p, t] for p in m.P for t in m.T))
    }


def clean(v, eps=1e-6):
    return 0.0 if abs(v) < eps else v


def print_plan(m: pyo.ConcreteModel):
    print("\nProd | Per | IndDem | RelMake | RelBuy | Receipt | EndInv | EndBacklog")
    print("-" * 78)
    for p in m.P:
        for t in m.T:
            print(
                f"{p:>4} | {t:>3} | "
                f"{clean(pyo.value(m.d[p, t])):>6.1f} | "
                f"{clean(pyo.value(m.r_make[p, t])):>7.1f} | "
                f"{clean(pyo.value(m.r_buy[p, t])):>6.1f} | "
                f"{clean(pyo.value(m.x[p, t])):>7.1f} | "
                f"{clean(pyo.value(m.I[p, t])):>6.1f} | "
                f"{clean(pyo.value(m.B[p, t])):>10.1f}"
            )
    print("-" * 78)

def print_plan_pivot(m: pyo.ConcreteModel):
    def clean(v, eps=1e-6):
        return 0.0 if abs(v) < eps else round(v, 1)

    periods = list(m.T)

    for p in m.P:
        print("\n" + p)
        print("".ljust(14), end="")
        for t in periods:
            print(f"{t:>8}", end="")
        print()

        # Demand
        print("demand".ljust(14), end="")
        for t in periods:
            print(f"{clean(pyo.value(m.d[p, t])):>8}", end="")
        print()

        # MAKE release
        print("make rel".ljust(14), end="")
        for t in periods:
            print(f"{clean(pyo.value(m.r_make[p, t])):>8}", end="")
        print()

        # BUY release
        print("buy rel".ljust(14), end="")
        for t in periods:
            print(f"{clean(pyo.value(m.r_buy[p, t])):>8}", end="")
        print()

        # Total receipts (after lead time)
        print("receipt".ljust(14), end="")
        for t in periods:
            print(f"{clean(pyo.value(m.x[p, t])):>8}", end="")
        print()

        # SOH
        print("SOH".ljust(14), end="")
        for t in periods:
            print(f"{clean(pyo.value(m.I[p, t])):>8}", end="")
        print()

        # Backlog (only if any)
        if any(clean(pyo.value(m.B[p, t])) > 0 for t in periods):
            print("backlog".ljust(14), end="")
            for t in periods:
                print(f"{clean(pyo.value(m.B[p, t])):>8}", end="")
            print()


if __name__ == "__main__":
    data = load_data("data.json")
    model = build_mrp_model(data)
    metrics = solve_three_phase(model)
    print(metrics)
    print_plan_pivot(model)

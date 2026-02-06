import argparse
import json
import os
from typing import Dict, Any, Iterable

import pandas as pd

import pyomo.environ as pyo

from mrp_model import MRPData, build_mrp_model


def build_mrpdata(raw: Dict[str, Any]) -> MRPData:
    periods = raw["periods"]
    locations = raw.get("locations") or ["LOC1"]

    def wrap_product_period_map(m):
        if not m:
            return {}
        sample = next(iter(m.values()))
        if isinstance(sample, dict) and any(k in periods for k in sample.keys()):
            # old format: product -> period -> value
            return {p: {locations[0]: v} for p, v in m.items()}
        return m

    def wrap_product_scalar_map(m):
        if not m:
            return {}
        sample = next(iter(m.values()))
        if isinstance(sample, (int, float)):
            return {p: {locations[0]: float(v)} for p, v in m.items()}
        return m

    def wrap_product_location_proc_map(m):
        if not m:
            return {}
        sample = next(iter(m.values()))
        if isinstance(sample, str):
            return {p: {locations[0]: str(v)} for p, v in m.items()}
        return m

    def wrap_product_location_int_map(m, default_val):
        if not m:
            return {}
        sample = next(iter(m.values()))
        if isinstance(sample, int):
            return {p: {locations[0]: int(v)} for p, v in m.items()}
        return m

    def wrap_product_location_float_map(m):
        if not m:
            return {}
        sample = next(iter(m.values()))
        if isinstance(sample, (int, float)):
            return {p: {locations[0]: float(v)} for p, v in m.items()}
        return m

    def wrap_bom_map(m):
        if not m:
            return {}
        sample = next(iter(m.values()))
        if isinstance(sample, dict) and any(isinstance(v, (int, float)) for v in sample.values()):
            # old format: parent -> component -> qty
            return {p: {locations[0]: v} for p, v in m.items()}
        return m

    return MRPData(
        products=raw["products"],
        periods=periods,
        locations=locations,

        demand=wrap_product_period_map(raw.get("demand", {})),
        initial_inventory=wrap_product_scalar_map(raw.get("initial_inventory", {})),
        capacity=wrap_product_period_map(raw.get("capacity", {})),
        cap_buy=wrap_product_period_map(raw.get("cap_buy", {})),
        bom=wrap_bom_map(raw.get("bom", {})),

        proc_type=wrap_product_location_proc_map(raw.get("proc_type", {})),

        lt_make=wrap_product_location_int_map(raw.get("lt_make", {}), 0),
        lt_buy=wrap_product_location_int_map(raw.get("lt_buy", {}), 0),

        min_lot_make=wrap_product_location_float_map(raw.get("min_lot_make", {})),
        multiple_lot_make=wrap_product_location_int_map(raw.get("multiple_lot_make", {}), 1),
        min_lot_buy=wrap_product_location_float_map(raw.get("min_lot_buy", {})),
        multiple_lot_buy=wrap_product_location_int_map(raw.get("multiple_lot_buy", {}), 1),

        ship_allowed=raw.get("ship_allowed", {}),
        ship_priority=raw.get("ship_priority", {}),
        ship_cap=raw.get("ship_cap", {}),
        lt_ship=raw.get("lt_ship", {}),

        allow_backlog=bool(raw.get("allow_backlog", True)),
    )


def load_data(path: str) -> MRPData:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)
    return build_mrpdata(raw)


def _df_to_dict(df: pd.DataFrame, keys: Iterable[str], value_col: str = "value") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for row in df.to_dict(orient="records"):
        cur = out
        for k in keys[:-1]:
            cur = cur.setdefault(str(row[k]), {})
        cur[str(row[keys[-1]])] = row[value_col]
    return out


def read_excel_to_raw(path: str) -> Dict[str, Any]:
    xls = pd.ExcelFile(path)

    def read_list(sheet: str, col: str) -> list[str]:
        df = pd.read_excel(xls, sheet_name=sheet)
        return [str(v) for v in df[col].dropna().tolist()]

    raw: Dict[str, Any] = {}
    raw["products"] = read_list("products", "product")
    raw["periods"] = read_list("periods", "period")
    raw["locations"] = read_list("locations", "location")

    if "demand" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="demand")
        raw["demand"] = _df_to_dict(df, ["product", "location", "period"])

    if "initial_inventory" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="initial_inventory")
        raw["initial_inventory"] = _df_to_dict(df, ["product", "location"])

    if "capacity" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="capacity")
        raw["capacity"] = _df_to_dict(df, ["product", "location", "period"])

    if "cap_buy" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="cap_buy")
        raw["cap_buy"] = _df_to_dict(df, ["product", "location", "period"])

    if "bom" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="bom").rename(columns={"component": "comp"})
        raw["bom"] = _df_to_dict(df, ["parent", "location", "comp"])

    if "proc_type" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="proc_type")
        raw["proc_type"] = _df_to_dict(df, ["product", "location"], value_col="proc_type")

    for sheet, key in [
        ("lt_make", "lt_make"),
        ("lt_buy", "lt_buy"),
        ("min_lot_make", "min_lot_make"),
        ("multiple_lot_make", "multiple_lot_make"),
        ("min_lot_buy", "min_lot_buy"),
        ("multiple_lot_buy", "multiple_lot_buy"),
    ]:
        if sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            raw[key] = _df_to_dict(df, ["product", "location"])

    if "ship_allowed" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="ship_allowed")
        raw["ship_allowed"] = _df_to_dict(df, ["product", "from", "to"], value_col="allowed")

    if "ship_priority" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="ship_priority")
        raw["ship_priority"] = _df_to_dict(df, ["product", "from", "to"], value_col="priority")

    if "lt_ship" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="lt_ship")
        raw["lt_ship"] = _df_to_dict(df, ["product", "from", "to"])

    if "ship_cap" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="ship_cap")
        raw["ship_cap"] = _df_to_dict(df, ["product", "from", "to", "period"])

    raw["allow_backlog"] = True
    return raw


def write_output_excel(path: str, m: pyo.ConcreteModel):
    periods = list(m.T)

    def row_values(values):
        return [clean(v) for v in values]

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # terminal-style sheet: key figures as rows, periods as columns
        rows = []
        first_group = True
        for p in m.P:
            for l in m.L:
                ship_out = [sum(pyo.value(m.ship[p, l, lt, t]) for lt in m.L) for t in m.T]
                ship_in = [sum(pyo.value(m.ship_receipt[p, lf, l, t]) for lf in m.L) for t in m.T]
                tot_dem = [
                    pyo.value(m.d_ind[p, l, t]) + pyo.value(m.d_dep[p, l, t]) + ship_out[i]
                    for i, t in enumerate(m.T)
                ]
                tot_rcpt = [
                    pyo.value(m.x[p, l, t]) + ship_in[i]
                    for i, t in enumerate(m.T)
                ]

                data = [
                    ("demand", row_values([pyo.value(m.d_ind[p, l, t]) for t in m.T])),
                    ("dep demand", row_values([pyo.value(m.d_dep[p, l, t]) for t in m.T])),
                    ("ship out", row_values(ship_out)),
                    ("total demand", row_values(tot_dem)),
                    ("make rel", row_values([pyo.value(m.r_make[p, l, t]) for t in m.T])),
                    ("buy rel", row_values([pyo.value(m.r_buy[p, l, t]) for t in m.T])),
                    ("ship in", row_values(ship_in)),
                    ("total receipts", row_values(tot_rcpt)),
                    ("SOH", row_values([pyo.value(m.I[p, l, t]) for t in m.T])),
                ]
                if not first_group:
                    rows.append({
                        "product": "",
                        "location": "",
                        "keyfigure": "",
                        **{per: "" for per in periods}
                    })
                first_group = False

                for keyfig, vals in data:
                    rows.append({
                        "product": p,
                        "location": l,
                        "keyfigure": keyfig,
                        **{periods[i]: vals[i] for i in range(len(periods))}
                    })

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="result", index=False)


def solve_three_phase(m: pyo.ConcreteModel, solver_name: str = "highs"):
    solver = pyo.SolverFactory(solver_name)
    if solver is None or not solver.available():
        raise RuntimeError("HiGHS solver not available (pip install highspy)")

    # Phase A: minimize backlog
    if hasattr(m, "Obj"):
        m.del_component(m.Obj)

    m.Obj = pyo.Objective(
        expr=sum(m.B[p, l, t] for p in m.P for l in m.L for t in m.T),
        sense=pyo.minimize
    )
    solver.solve(m)

    best_backlog = pyo.value(sum(m.B[p, l, t] for p in m.P for l in m.L for t in m.T))

    # Phase B: minimize inventory
    m.KeepBacklog = pyo.Constraint(
        expr=sum(m.B[p, l, t] for p in m.P for l in m.L for t in m.T) <= best_backlog + 1e-6
    )

    m.del_component(m.Obj)
    m.Obj = pyo.Objective(
        expr=sum(m.I[p, l, t] for p in m.P for l in m.L for t in m.T),
        sense=pyo.minimize
    )
    solver.solve(m)

    best_inventory = pyo.value(sum(m.I[p, l, t] for p in m.P for l in m.L for t in m.T))

    # Phase C: minimize BUY (MAKE preferred)
    m.KeepInventory = pyo.Constraint(
        expr=sum(m.I[p, l, t] for p in m.P for l in m.L for t in m.T) <= best_inventory + 1e-6
    )

    m.del_component(m.Obj)
    m.Obj = pyo.Objective(
        expr=sum(m.r_buy[p, l, t] for p in m.P for l in m.L for t in m.T),
        sense=pyo.minimize
    )
    solver.solve(m)

    best_buy = pyo.value(sum(m.r_buy[p, l, t] for p in m.P for l in m.L for t in m.T))

    # Phase D: minimize transfer priority (lower priority value preferred)
    m.KeepBuy = pyo.Constraint(
        expr=sum(m.r_buy[p, l, t] for p in m.P for l in m.L for t in m.T) <= best_buy + 1e-6
    )

    def ship_priority_weight(mm, p, lf, lt, t):
        return (mm.ship_priority[p, lf, lt] + 1) * mm.ship[p, lf, lt, t]

    m.del_component(m.Obj)
    m.Obj = pyo.Objective(
        expr=sum(ship_priority_weight(m, p, lf, lt, t) for p in m.P for lf in m.L for lt in m.L for t in m.T),
        sense=pyo.minimize
    )
    solver.solve(m)

    return {
        "backlog": best_backlog,
        "inventory": best_inventory,
        "buy_volume": best_buy
    }


def clean(v, eps=1e-6):
    return 0.0 if abs(v) < eps else v


def print_plan(m: pyo.ConcreteModel):
    print("\nProd | Loc | Per | IndDem | DepDem | TotDem | RelMake | RelBuy | ShipOut | ShipIn | TotalRcpt | EndInv | EndBacklog")
    print("-" * 78)
    for p in m.P:
        for l in m.L:
            for t in m.T:
                ship_out = sum(pyo.value(m.ship[p, l, lt, t]) for lt in m.L)
                ship_in = sum(pyo.value(m.ship_receipt[p, lf, l, t]) for lf in m.L)
                tot_dem = pyo.value(m.d_ind[p, l, t]) + pyo.value(m.d_dep[p, l, t]) + ship_out
                tot_rcpt = pyo.value(m.x[p, l, t]) + ship_in
                print(
                    f"{p:>4} | {l:>3} | {t:>3} | "
                    f"{clean(pyo.value(m.d_ind[p, l, t])):>6.1f} | "
                    f"{clean(pyo.value(m.d_dep[p, l, t])):>6.1f} | "
                    f"{clean(tot_dem):>6.1f} | "
                    f"{clean(pyo.value(m.r_make[p, l, t])):>7.1f} | "
                    f"{clean(pyo.value(m.r_buy[p, l, t])):>6.1f} | "
                    f"{clean(ship_out):>7.1f} | "
                    f"{clean(ship_in):>6.1f} | "
                    f"{clean(tot_rcpt):>9.1f} | "
                    f"{clean(pyo.value(m.I[p, l, t])):>6.1f} | "
                    f"{clean(pyo.value(m.B[p, l, t])):>10.1f}"
                )
    print("-" * 78)

def print_plan_pivot(m: pyo.ConcreteModel):
    def clean(v, eps=1e-6):
        return 0.0 if abs(v) < eps else round(v, 1)

    periods = list(m.T)

    for p in m.P:
        for l in m.L:
            print("\n" + f"{p} @ {l}")
            print("".ljust(14), end="")
            for t in periods:
                print(f"{t:>8}", end="")
            print()

            # 1. demand
            print("demand".ljust(14), end="")
            for t in periods:
                print(f"{clean(pyo.value(m.d_ind[p, l, t])):>8}", end="")
            print()

            # 2. dep demand
            print("dep demand".ljust(14), end="")
            for t in periods:
                print(f"{clean(pyo.value(m.d_dep[p, l, t])):>8}", end="")
            print()

            # 3. ship out
            print("ship out".ljust(14), end="")
            for t in periods:
                ship_out = sum(pyo.value(m.ship[p, l, lt, t]) for lt in m.L)
                print(f"{clean(ship_out):>8}", end="")
            print()

            # 4. total demand (ind + dep + ship out)
            print("total demand".ljust(14), end="")
            for t in periods:
                ship_out = sum(pyo.value(m.ship[p, l, lt, t]) for lt in m.L)
                tot_dem = pyo.value(m.d_ind[p, l, t]) + pyo.value(m.d_dep[p, l, t]) + ship_out
                print(f"{clean(tot_dem):>8}", end="")
            print()

            # 5. make rel
            print("make rel".ljust(14), end="")
            for t in periods:
                print(f"{clean(pyo.value(m.r_make[p, l, t])):>8}", end="")
            print()

            # 6. buy rel
            print("buy rel".ljust(14), end="")
            for t in periods:
                print(f"{clean(pyo.value(m.r_buy[p, l, t])):>8}", end="")
            print()

            # 7. ship in
            print("ship in".ljust(14), end="")
            for t in periods:
                ship_in = sum(pyo.value(m.ship_receipt[p, lf, l, t]) for lf in m.L)
                print(f"{clean(ship_in):>8}", end="")
            print()

            # 8. total receipts (after lead time + ship in)
            print("total receipts".ljust(14), end="")
            for t in periods:
                tot_rcpt = pyo.value(m.x[p, l, t]) + sum(pyo.value(m.ship_receipt[p, lf, l, t]) for lf in m.L)
                print(f"{clean(tot_rcpt):>8}", end="")
            print()

            # 9. SOH
            print("SOH".ljust(14), end="")
            for t in periods:
                print(f"{clean(pyo.value(m.I[p, l, t])):>8}", end="")
            print()

            # Backlog (only if any)
            if any(clean(pyo.value(m.B[p, l, t])) > 0 for t in periods):
                print("backlog".ljust(14), end="")
                for t in periods:
                    print(f"{clean(pyo.value(m.B[p, l, t])):>8}", end="")
                print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel-input", dest="excel_input", default="input.xlsx", help="Path to input Excel file")
    parser.add_argument("--json-output", dest="json_output", default="data.json", help="Path to write data.json")
    parser.add_argument("--excel-output", dest="excel_output", default="mrp_result.xlsx", help="Path to output Excel file")
    parser.add_argument("--json-input", dest="json_input", default="data.json", help="Path to input data.json")
    parser.add_argument("--no-open", dest="no_open", action="store_true", help="Do not open Excel files after run")
    args = parser.parse_args()

    try:
        raw = read_excel_to_raw(args.excel_input)
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)
        data = build_mrpdata(raw)
    except FileNotFoundError:
        data = load_data(args.json_input)

    model = build_mrp_model(data)
    metrics = solve_three_phase(model)
    print(metrics)
    print_plan_pivot(model)

    write_output_excel(args.excel_output, model)

    # Open input/output files (best-effort)
    if not args.no_open:
        for path in [args.excel_input, args.excel_output]:
            if path and os.path.exists(path):
                try:
                    os.startfile(path)
                except Exception:
                    pass

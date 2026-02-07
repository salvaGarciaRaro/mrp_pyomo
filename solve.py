import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, Iterable

import pandas as pd

import pyomo.environ as pyo

from mrp_model import MRPData, build_mrp_model


def _apply_bom_rows(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "bom_rows" not in raw:
        return raw

    bom: Dict[str, Dict[str, Dict[str, float]]] = {}
    lt_make = raw.get("lt_make", {})
    min_lot_make = raw.get("min_lot_make", {})
    mult_lot_make = raw.get("multiple_lot_make", {})

    for row in raw.get("bom_rows", []):
        parent = str(row.get("parent", "")).strip()
        location = str(row.get("location", "")).strip()
        component = str(row.get("component", "")).strip()
        if not parent or not location or not component:
            continue
        qty = row.get("value", row.get("qty", 0))
        bom.setdefault(parent, {}).setdefault(location, {})[component] = float(qty)

        if "lt_make" in row and row["lt_make"] is not None:
            lt_make.setdefault(parent, {})[location] = int(row["lt_make"])
        if "min_lot_make" in row and row["min_lot_make"] is not None:
            min_lot_make.setdefault(parent, {})[location] = float(row["min_lot_make"])
        if "multiple_lot_make" in row and row["multiple_lot_make"] is not None:
            mult_lot_make.setdefault(parent, {})[location] = int(row["multiple_lot_make"])

    raw["bom"] = bom
    if lt_make:
        raw["lt_make"] = lt_make
    if min_lot_make:
        raw["min_lot_make"] = min_lot_make
    if mult_lot_make:
        raw["multiple_lot_make"] = mult_lot_make
    return raw


def _apply_ship_rows(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "ship_rows" not in raw:
        return raw

    ship_allowed: Dict[str, Dict[str, Dict[str, bool]]] = {}
    ship_priority: Dict[str, Dict[str, Dict[str, int]]] = {}
    lt_ship: Dict[str, Dict[str, Dict[str, int]]] = {}

    for row in raw.get("ship_rows", []):
        prod = str(row.get("product", "")).strip()
        lf = str(row.get("from", "")).strip()
        lt = str(row.get("to", "")).strip()
        if not prod or not lf or not lt:
            continue
        allowed = row.get("allowed", True)
        ship_allowed.setdefault(prod, {}).setdefault(lf, {})[lt] = bool(allowed)
        if row.get("priority") is not None:
            ship_priority.setdefault(prod, {}).setdefault(lf, {})[lt] = int(row["priority"])
        if row.get("lt_ship") is not None:
            lt_ship.setdefault(prod, {}).setdefault(lf, {})[lt] = int(row["lt_ship"])

    raw["ship_allowed"] = ship_allowed
    if ship_priority:
        raw["ship_priority"] = ship_priority
    if lt_ship:
        raw["lt_ship"] = lt_ship
    return raw


def _apply_purchasing_rows(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "purchasing_rows" not in raw:
        return raw

    lt_buy = raw.get("lt_buy", {})
    min_lot_buy = raw.get("min_lot_buy", {})
    mult_lot_buy = raw.get("multiple_lot_buy", {})
    buy_defined: Dict[str, Dict[str, bool]] = raw.get("buy_defined", {})

    for row in raw.get("purchasing_rows", []):
        prod = str(row.get("product", "")).strip()
        loc = str(row.get("location", "")).strip()
        if not prod or not loc:
            continue
        buy_defined.setdefault(prod, {})[loc] = True
        if row.get("leadtime") is not None:
            lt_buy.setdefault(prod, {})[loc] = int(row["leadtime"])
        if row.get("min_lotsize") is not None:
            min_lot_buy.setdefault(prod, {})[loc] = float(row["min_lotsize"])
        if row.get("mult_lotsize") is not None:
            mult_lot_buy.setdefault(prod, {})[loc] = int(row["mult_lotsize"])

    if lt_buy:
        raw["lt_buy"] = lt_buy
    if min_lot_buy:
        raw["min_lot_buy"] = min_lot_buy
    if mult_lot_buy:
        raw["multiple_lot_buy"] = mult_lot_buy
    if buy_defined:
        raw["buy_defined"] = buy_defined
    return raw


def build_mrpdata(raw: Dict[str, Any]) -> MRPData:
    raw = _apply_bom_rows(raw)
    raw = _apply_ship_rows(raw)
    raw = _apply_purchasing_rows(raw)
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

    def wrap_resource_period_map(m):
        if not m:
            return {}
        sample = next(iter(m.values()))
        if isinstance(sample, dict) and any(k in periods for k in sample.keys()):
            # old format: resource -> period -> value
            return {r: {locations[0]: v} for r, v in m.items()}
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

    def wrap_product_location_bool_map(m):
        if not m:
            return {}
        sample = next(iter(m.values()))
        if isinstance(sample, bool):
            return {p: {locations[0]: bool(v)} for p, v in m.items()}
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
        resources=raw.get("resources", []),

        demand=wrap_product_period_map(raw.get("demand", {})),
        initial_inventory=wrap_product_scalar_map(raw.get("initial_inventory", {})),
        capacity=wrap_resource_period_map(raw.get("capacity", {})),
        cap_buy=wrap_product_period_map(raw.get("cap_buy", {})),
        bom=wrap_bom_map(raw.get("bom", {})),
        routing=raw.get("routing", {}),

        proc_type=wrap_product_location_proc_map(raw.get("proc_type", {})),

        lt_make=wrap_product_location_int_map(raw.get("lt_make", {}), 0),
        lt_buy=wrap_product_location_int_map(raw.get("lt_buy", {}), 0),

        min_lot_make=wrap_product_location_float_map(raw.get("min_lot_make", {})),
        multiple_lot_make=wrap_product_location_int_map(raw.get("multiple_lot_make", {}), 1),
        min_lot_buy=wrap_product_location_float_map(raw.get("min_lot_buy", {})),
        multiple_lot_buy=wrap_product_location_int_map(raw.get("multiple_lot_buy", {}), 1),
        buy_defined=wrap_product_location_bool_map(raw.get("buy_defined", {})),

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
    raw["products"] = read_list("Products", "product")
    raw["periods"] = read_list("Time_periods", "period")
    raw["locations"] = read_list("Locations", "location")
    if "Resources" in xls.sheet_names:
        raw["resources"] = read_list("Resources", "resource")

    if "Independent_Demand" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="Independent_Demand")
        raw["demand"] = _df_to_dict(df, ["product", "location", "period"])

    if "Initial_Inventory" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="Initial_Inventory")
        raw["initial_inventory"] = _df_to_dict(df, ["product", "location"])

    if "Resource_Capacity" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="Resource_Capacity")
        raw["capacity"] = _df_to_dict(df, ["resource", "location", "period"])

    if "Purchasing_Capacity" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="Purchasing_Capacity")
        raw["cap_buy"] = _df_to_dict(df, ["product", "location", "period"])

    if "BOM" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="BOM")
        # Keep JSON aligned with Excel: store BOM rows directly (including lt/make lots)
        bom_rows = []
        for row in df.to_dict(orient="records"):
            rec = {
                "parent": row.get("parent"),
                "location": row.get("location"),
                "component": row.get("component"),
                "value": row.get("value", row.get("qty")),
            }
            if "lt_make" in df.columns:
                rec["lt_make"] = row.get("lt_make")
            if "min_lot_make" in df.columns:
                rec["min_lot_make"] = row.get("min_lot_make")
            if "multiple_lot_make" in df.columns:
                rec["multiple_lot_make"] = row.get("multiple_lot_make")
            bom_rows.append(rec)
        raw["bom_rows"] = bom_rows

    if "Routing" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="Routing")
        raw["routing"] = _df_to_dict(df, ["product", "location", "resource"])

    if "Proc_Type" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="Proc_Type")
        raw["proc_type"] = _df_to_dict(df, ["product", "location"], value_col="proc_type")

    if "Purchasing" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="Purchasing")
        purchasing_rows = []
        for row in df.to_dict(orient="records"):
            purchasing_rows.append({
                "product": row.get("product"),
                "location": row.get("location"),
                "leadtime": row.get("leadtime"),
                "min_lotsize": row.get("min lotsize"),
                "mult_lotsize": row.get("mult lotsize"),
            })
        raw["purchasing_rows"] = purchasing_rows

    if "TransportationLanes" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="TransportationLanes")
        ship_rows = []
        for row in df.to_dict(orient="records"):
            rec = {
                "product": row.get("product"),
                "from": row.get("from"),
                "to": row.get("to"),
                "allowed": row.get("allowed", True),
            }
            if "priority" in df.columns:
                rec["priority"] = row.get("priority")
            if "lt_ship" in df.columns:
                rec["lt_ship"] = row.get("lt_ship")
            ship_rows.append(rec)
        raw["ship_rows"] = ship_rows

    if "Transportation_Capacity" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="Transportation_Capacity")
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
                    ("independent demand", row_values([pyo.value(m.d_ind[p, l, t]) for t in m.T])),
                    ("dep demand", row_values([pyo.value(m.d_dep[p, l, t]) for t in m.T])),
                    ("distribution req", row_values(ship_out)),
                    ("total demand", row_values(tot_dem)),
                    ("production receipts", row_values([pyo.value(m.r_make[p, l, t]) for t in m.T])),
                    ("procurement receipts", row_values([pyo.value(m.r_buy[p, l, t]) for t in m.T])),
                    ("distribution rec", row_values(ship_in)),
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

        # Resource consumption (aggregate)
        res_rows = []
        for r in m.R:
            for l in m.L:
                vals = []
                for t in m.T:
                    cons = sum(
                        pyo.value(m.r_make[p, l, t]) * pyo.value(m.routing[p, l, r])
                        for p in m.P
                    )
                    vals.append(clean(cons))
                res_rows.append({
                    "resource": r,
                    "location": l,
                    **{periods[i]: vals[i] for i in range(len(periods))}
                })
        pd.DataFrame(res_rows).to_excel(writer, sheet_name="Resource_Consumption", index=False)

        # Resource consumption breakdown by product/location
        detail_rows = []
        for r in m.R:
            for l in m.L:
                for p in m.P:
                    vals = []
                    for t in m.T:
                        cons = pyo.value(m.r_make[p, l, t]) * pyo.value(m.routing[p, l, r])
                        vals.append(clean(cons))
                    if any(v != 0.0 for v in vals):
                        detail_rows.append({
                            "resource": r,
                            "location": l,
                            "product": p,
                            **{periods[i]: vals[i] for i in range(len(periods))}
                        })
        pd.DataFrame(detail_rows).to_excel(writer, sheet_name="Resource_Consumption_Detail", index=False)


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

    # Phase B: minimize BUY (prefer transfer over buy when make not possible)
    m.KeepBacklog = pyo.Constraint(
        expr=sum(m.B[p, l, t] for p in m.P for l in m.L for t in m.T) <= best_backlog + 1e-6
    )

    m.del_component(m.Obj)
    m.Obj = pyo.Objective(
        expr=sum(m.r_buy[p, l, t] for p in m.P for l in m.L for t in m.T),
        sense=pyo.minimize
    )
    solver.solve(m)

    best_buy = pyo.value(sum(m.r_buy[p, l, t] for p in m.P for l in m.L for t in m.T))

    # Phase C: minimize inventory (given best backlog + buy)
    m.KeepBuy = pyo.Constraint(
        expr=sum(m.r_buy[p, l, t] for p in m.P for l in m.L for t in m.T) <= best_buy + 1e-6
    )

    m.del_component(m.Obj)
    m.Obj = pyo.Objective(
        expr=sum(m.I[p, l, t] for p in m.P for l in m.L for t in m.T),
        sense=pyo.minimize
    )
    solver.solve(m)

    best_inventory = pyo.value(sum(m.I[p, l, t] for p in m.P for l in m.L for t in m.T))

    # Phase D: minimize transfer priority (lower priority value preferred)
    m.KeepInventory = pyo.Constraint(
        expr=sum(m.I[p, l, t] for p in m.P for l in m.L for t in m.T) <= best_inventory + 1e-6
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

            # 1. independent demand
            print("independent demand".ljust(18), end="")
            for t in periods:
                print(f"{clean(pyo.value(m.d_ind[p, l, t])):>8}", end="")
            print()

            # 2. dep demand
            print("dep demand".ljust(18), end="")
            for t in periods:
                print(f"{clean(pyo.value(m.d_dep[p, l, t])):>8}", end="")
            print()

            # 3. ship out
            print("distribution req".ljust(18), end="")
            for t in periods:
                ship_out = sum(pyo.value(m.ship[p, l, lt, t]) for lt in m.L)
                print(f"{clean(ship_out):>8}", end="")
            print()

            # 4. total demand (ind + dep + ship out)
            print("total demand".ljust(18), end="")
            for t in periods:
                ship_out = sum(pyo.value(m.ship[p, l, lt, t]) for lt in m.L)
                tot_dem = pyo.value(m.d_ind[p, l, t]) + pyo.value(m.d_dep[p, l, t]) + ship_out
                print(f"{clean(tot_dem):>8}", end="")
            print()

            # 5. make rel
            print("production receipts".ljust(18), end="")
            for t in periods:
                print(f"{clean(pyo.value(m.r_make[p, l, t])):>8}", end="")
            print()

            # 6. buy rel
            print("procurement receipts".ljust(18), end="")
            for t in periods:
                print(f"{clean(pyo.value(m.r_buy[p, l, t])):>8}", end="")
            print()

            # 7. ship in
            print("distribution rec".ljust(18), end="")
            for t in periods:
                ship_in = sum(pyo.value(m.ship_receipt[p, lf, l, t]) for lf in m.L)
                print(f"{clean(ship_in):>8}", end="")
            print()

            # 8. total receipts (after lead time + ship in)
            print("total receipts".ljust(18), end="")
            for t in periods:
                tot_rcpt = pyo.value(m.x[p, l, t]) + sum(pyo.value(m.ship_receipt[p, lf, l, t]) for lf in m.L)
                print(f"{clean(tot_rcpt):>8}", end="")
            print()

            # 9. SOH
            print("SOH".ljust(18), end="")
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

    # If output is default, add timestamp to avoid overwrite
    if args.excel_output == "mrp_result.xlsx":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.excel_output = f"mrp_result_{ts}.xlsx"

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

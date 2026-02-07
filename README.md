# MRP Pyomo Model

This project implements a multi‑product, multi‑location MRP model in Pyomo.  
It supports make, buy, and transfer decisions, BOM‑driven dependent demand, resource‑based capacity, lead times, and lot sizing.  
Inputs are provided in Excel, and outputs are written to Excel with a terminal‑style layout.

---

## How It Works (Functional Logic)

The optimization is solved in phases (lexicographic priorities):

1. **Minimize backlog**
2. **Minimize buy volume** (prefers transfer over buy when make isn’t possible)
3. **Minimize total inventory**
4. **Minimize transfers by lane priority**

Transfers are only allowed for procurement types **F** or **X**.  
`P` = make only, `F` = buy/transfer only, `X` = make/buy/transfer.

---

## Inputs

The default input file is `input.xlsx`.  
If the file is missing, the model reads `data.json` instead.

### Excel Tabs (required names & order)

1. `Time_periods`
2. `Locations`
3. `Products`
4. `Proc_Type`
5. `Purchasing_Capacity`
6. `Resources`
7. `Resource_Capacity`
8. `BOM`
9. `Routing`
10. `Purchasing`
11. `TransportationLanes`
12. `Transportation_Capacity`
13. `Initial_Inventory`
14. `Independent_Demand`

### Key tab notes

**BOM**
- Columns: `parent`, `location`, `component`, `value`, `lt_make`, `min_lot_make`, `multiple_lot_make`

**Purchasing**
- Columns: `product`, `location`, `leadtime`, `min lotsize`, `mult lotsize`
- Buying is allowed **only** for product/location rows present here.

**TransportationLanes**
- Columns: `product`, `from`, `to`, `allowed`, `priority`, `lt_ship`

**Resource_Capacity**
- Columns: `resource`, `location`, `period`, `value`

**Purchasing_Capacity**
- Columns: `product`, `location`, `period`, `value`

---

## Outputs

The results are written to an Excel file (timestamped by default):

- `result`  
  Key figures as rows, periods as columns.

Key figures include:
- independent demand
- dep demand
- distribution req
- total demand
- production receipts
- procurement receipts
- distribution rec
- total receipts
- SOH

Additional output tabs:
- `Resource_Consumption`  
  Total resource usage by resource/location and period.
- `Resource_Consumption_Detail`  
  Product/location breakdown of resource usage.

---

## Run

```bash
python solve.py
```

Default behavior:
- Reads `input.xlsx`
- Falls back to `data.json` if Excel is missing
- Writes output to `mrp_result_<timestamp>.xlsx`

---

## Dependencies

```
pyomo
pandas
highspy
openpyxl
python-pptx
```

---

## Notes

- Transfers are only allowed if procurement type is `F` or `X`.
- Buying is only allowed if a row exists in `Purchasing`.

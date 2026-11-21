"""Reused V0.25 streaming UCI adapter (function unchanged)."""
from collections import defaultdict, Counter
from datetime import datetime, date
import pandas as pd
import numpy as np
import openpyxl

def load_uci_streaming(path, max_series=100):
    """Read both UCI workbook sheets in streaming mode; aggregate to product-country-Sunday-week."""
    wb=openpyxl.load_workbook(path,read_only=True,data_only=True)
    agg=defaultdict(lambda:[0.0,0.0,0])  # demand, revenue, positive line count
    quality=Counter()
    for ws in wb.worksheets:
        header=next(ws.iter_rows(min_row=1,max_row=1,values_only=True))
        idx={str(v).strip():i for i,v in enumerate(header) if v is not None}
        aliases={
            "invoice": next((idx[k] for k in ("Invoice","InvoiceNo") if k in idx),None),
            "stock": next((idx[k] for k in ("StockCode",) if k in idx),None),
            "quantity": next((idx[k] for k in ("Quantity",) if k in idx),None),
            "date": next((idx[k] for k in ("InvoiceDate",) if k in idx),None),
            "price": next((idx[k] for k in ("Price","UnitPrice") if k in idx),None),
            "country": next((idx[k] for k in ("Country",) if k in idx),None)
        }
        if any(v is None for v in aliases.values()):
            raise ValueError(f"Unexpected UCI workbook schema: {list(idx)}")
        for row in ws.iter_rows(min_row=2,values_only=True):
            quality["raw_rows"]+=1
            inv=row[aliases["invoice"]]; stock=row[aliases["stock"]]
            qty=row[aliases["quantity"]]; dte=row[aliases["date"]]
            price=row[aliases["price"]]; country=row[aliases["country"]]
            if inv is None or stock is None or dte is None or country is None:
                quality["missing_key_rows"]+=1; continue
            if str(inv).strip().upper().startswith("C"):
                quality["cancel_invoice_rows"]+=1; continue
            try: qty=float(qty); price=float(price)
            except (TypeError,ValueError):
                quality["non_numeric_rows"]+=1; continue
            if qty<=0:
                quality["nonpositive_quantity_rows"]+=1; continue
            if price<=0:
                quality["nonpositive_price_rows"]+=1; continue
            if isinstance(dte,datetime): dd=dte.date()
            elif isinstance(dte,date): dd=dte
            else:
                dd=pd.to_datetime(dte,dayfirst=True,errors="coerce")
                if pd.isna(dd):
                    quality["invalid_date_rows"]+=1; continue
                dd=dd.date()
            # W-SAT calendar: Sunday through Saturday, consistent with the adapter.
            week_start=dd-pd.Timedelta(days=(dd.weekday()+1)%7)
            key=(str(stock),str(country),week_start)
            rec=agg[key]; rec[0]+=qty; rec[1]+=qty*price; rec[2]+=1
            quality["valid_rows"]+=1
    wb.close()

    weeks=sorted({k[2] for k in agg})
    week_map={w:i for i,w in enumerate(weeks)}
    cutoff=weeks[int(len(weeks)*0.60)]
    train_totals=defaultdict(float)
    for (stock,country,w),(qty,rev,nlines) in agg.items():
        if w<=cutoff:
            train_totals[(stock,country)]+=qty
    top=set(sorted(train_totals,key=train_totals.get,reverse=True)[:max_series])
    rows=[]
    for (stock,country,w),(qty,rev,nlines) in agg.items():
        if (stock,country) not in top: continue
        rows.append({
            "series_id":stock+"::"+country,"time_index":week_map[w],"date":str(w),
            "demand":qty,"price":rev/max(qty,1e-12),
            "promotion":np.nan,"promotion_observed":0,"source":"uci_online_retail_ii"
        })
    panel=pd.DataFrame(rows).sort_values(["series_id","time_index"]).reset_index(drop=True)
    quality["aggregated_product_country_week_rows"]=len(agg)
    quality["distinct_product_country_series"]=len(train_totals)
    quality["calendar_weeks"]=len(weeks)
    quality["selected_series"]=len(top)
    quality["selected_panel_rows"]=len(panel)
    quality["top_series_selection_cutoff"]=str(cutoff)
    return panel,dict(quality)

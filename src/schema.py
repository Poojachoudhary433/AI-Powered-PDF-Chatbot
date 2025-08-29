from pydantic import BaseModel
from typing import List, Optional

# --- Define schema ---
class KeyValue(BaseModel):
    key: str
    value: str

class BillParsedData(BaseModel):
    header: List[KeyValue]
    seller_info: List[KeyValue]
    buyer_info: List[KeyValue]
    totals: List[KeyValue]
    line_items: List[dict]   # Keep it generic for line items
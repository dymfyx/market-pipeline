from datetime import date
from pydantic import BaseModel, ConfigDict

class StockDaily(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    date: date
    open: float 
    high: float
    low: float
    close: float    
    volume: float
    amount: float
    outstanding_share: float
    turnover: float
    

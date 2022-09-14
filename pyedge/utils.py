from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel


class Message(BaseModel):
    frames: List[np.ndarray]
    timestamp: datetime
    predictions: Optional[List[Dict]]

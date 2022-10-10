from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel


class Message(BaseModel, arbitrary_types_allowed=True):
    frames: List[np.ndarray]
    timestamp: datetime
    predictions: Optional[List[Dict]]

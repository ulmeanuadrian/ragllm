from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict

class CollectionCreate(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    description: Optional[str] = None

class DocumentDeleteRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    document_ids: List[str]
    
class GenerateRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    query: str
    top_k_docs: Optional[int] = 5
    temperature: Optional[float] = 0.2
    use_web_search: Optional[bool] = False

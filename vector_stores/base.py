from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def similarity_search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass

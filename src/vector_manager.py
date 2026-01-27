import os
import shutil
from typing import List
from src.config import AppConfig

class VectorStoreManager:
    """Manages multiple Vector Databases."""

    def __init__(self):
        self.base_dir = AppConfig.VECTOR_DB_DIR
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def list_dbs(self) -> List[str]:
        """Returns a list of available vector database names."""
        if not os.path.exists(self.base_dir):
            return []
        
        dbs = [
            d for d in os.listdir(self.base_dir) 
            if os.path.isdir(os.path.join(self.base_dir, d))
        ]
        return sorted(dbs)

    def get_db_path(self, db_name: str) -> str:
        """Returns the absolute path for a given DB name."""
        return os.path.join(self.base_dir, db_name)

    def create_db_dir(self, db_name: str) -> str:
        """Creates a directory for a new DB."""
        path = self.get_db_path(db_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    
    def delete_db(self, db_name: str):
        """Deletes a vector database."""
        path = self.get_db_path(db_name)
        if os.path.exists(path):
            shutil.rmtree(path)

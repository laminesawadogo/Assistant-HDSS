import sys
from pathlib import Path

# Permet d'importer data_tools.py et rag.py depuis le dossier parent des tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

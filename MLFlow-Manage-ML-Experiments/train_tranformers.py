# test_transformers_import.py
try:
    from transformers import BertTokenizer, BertModel
    print("Transformers importado correctamente.")
except ImportError as e:
    print(f"Error al importar transformers: {e}")

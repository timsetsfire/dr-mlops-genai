import os 
import pandas as pd
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import logging 
import torch
import json

logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(
                format="{} - %(levelname)s - %(asctime)s - %(message)s".format("debug-logger"),
        )
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

def load_model(code_dir):
    """
    This hook can be implemented to adjust logic in the scoring mode.

    load_model hook provides a way to implement model loading your self.
    This function should return an object that represents your model. This object will
    be passed to the predict hook for performing predictions.
    This hook can be used to load supported models if your model has multiple artifacts, or
    for loading models that drum does not natively support

    :param code_dir: the directory to load serialized models from
    :returns: Object containing the model - the predict hook will get this object as a parameter
    """
    if not torch.cuda.is_available():
        EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    else:
        EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
    # Returning a string with value "dummy" as the model.
    embedding_function = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder= os.path.join(code_dir, "sentencetransformers"),
    )
    db = FAISS.load_local( os.path.join(code_dir, "faiss_index"), embedding_function, allow_dangerous_deserialization=True)
    return db.as_retriever()

def score_unstructured(model, data, query, **kwargs):
    headers = kwargs.get("headers")
    logger.info("="*20)
    logger.info("headers")
    logger.info(headers)
    logger.info("="*20)
    logger.info("data")
    logger.info(data)
    logger.info("="*20)
    docs = model.get_relevant_documents(data)
    return json.dumps([doc.__dict__ for doc in docs])
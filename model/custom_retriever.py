# from langchain.schema.retriever import BaseRetriever, Document
from langchain_core.retrievers import BaseRetriever 
from langchain.docstore.document import Document

from typing import TYPE_CHECKING, Any, Dict, List, Optional 
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from datarobot_drum import RuntimeParameters
import requests
import logging

class CustomRetriever(BaseRetriever):
    api_token_key:str
    api_end_point:str
    api_headers: dict

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """
        _get_relevant_documents is function of BaseRetriever implemented here

        :param query: String value of the query

        """
        result_docs = list()

        response = requests.post(self.api_end_point,
                                 headers=self.api_headers,
                                 data=query)
        
        docs = response.json()
        logging.info(type(docs))
        logging.info(docs)

        docs = [Document(page_content = d["page_content"], metadata = d["metadata"], type = d["type"]) for d in docs]
        return docs
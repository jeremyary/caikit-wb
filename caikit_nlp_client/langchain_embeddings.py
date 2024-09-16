import json
from typing import List
from langchain_core.embeddings.embeddings import Embeddings
from caikit_nlp_client import HttpClient

class LangchainEmbeddings(Embeddings):
    def __init__(self, token: str, endpoint: str, verify: bool = False, model: str = "all-MiniLM-L12-v2-caikit") -> None:
        self.client = HttpClient(base_url=endpoint, verify=verify)
        self.token = token
        self.endpoint = endpoint
        self.verify = verify
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Synchronously embed a list of documents."""
        response = self.client.embedding_tasks(
            token=self.token,
            model_id=self.model,
            texts=texts,
        )

        return_list = []
        list_from_response = list(response)
        for i in range(len(list_from_response)):
            if type(list_from_response[i]) is dict and ("vectors" in (list_from_response[i])):
                pulled_vector_list = list_from_response[i]["vectors"]
                for j in range(len(pulled_vector_list)):
                    veclist = pulled_vector_list[j]["data"]["values"]
                    return_list.append(veclist)

        return return_list

    def embed_query(self, text: str) -> List[float]:
        """Synchronously embed a single query."""
        response = self.client.embedding(
            token=self.token,
            model_id=self.model,
            text=text
        )
        print(f"single-end::{type(response)}")
        return response
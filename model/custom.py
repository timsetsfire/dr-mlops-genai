import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from custom_retriever import * 
from langchain.chains import create_retrieval_chain

# open

DATAROBOT_API_TOKEN = RuntimeParameters.get("DATAROBOT_API_TOKEN")["apiToken"]
OPENAI_API_KEY = RuntimeParameters.get("OPENAI_API_KEY")["apiToken"]
VDB_URL = RuntimeParameters.get("VDB_URL")
DATAROBOT_KEY = RuntimeParameters.get("DATAROBOT_KEY")["apiToken"]
HEADERS = {
    'Authorization': f'Bearer {DATAROBOT_API_TOKEN}',
    'DataRobot-Key': DATAROBOT_KEY,
    "Content-Type": "text/plain"
    }

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

    # Returning a string with value "dummy" as the model.
    retriever = CustomRetriever(api_token_key = DATAROBOT_API_TOKEN, 
                                api_end_point = VDB_URL,
                                api_headers = HEADERS)
    model = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        temperature=0.4,api_key=OPENAI_API_KEY
    )

    prompt = ChatPromptTemplate.from_template("""
        Answer the user's question:
        Context: {context}
        Question: {input}                                       
    """)

    # chain = prompt | model
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retrieval_chain

def score(data, model, **kwargs):
    prompts = data["promptText"].tolist()
    responses = []
    for prompt in prompts:
        response = model.invoke({"input": prompt})
        responses.append( response["answer"] )
    return pd.DataFrame( dict( resultText = responses))
    
    

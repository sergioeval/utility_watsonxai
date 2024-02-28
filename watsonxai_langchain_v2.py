import requests 
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
#import pprint
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from dotenv import load_dotenv
import os
#import time
load_dotenv()

# env variables
IBM_CLOUD_API_KEY = os.getenv(key='IBM_CLOUD_API_KEY')
IBM_ML_ENDPOINT_URL = os.getenv(key='IBM_ML_ENDPOINT_URL')
WATSONXAI_PROJECT_ID = os.getenv(key='WATSONXAI_PROJECT_ID')

# this is a comment from code spaces.

class WatsonxAiLangchain:
    '''
        To make available watsonxai usage 
        Based on this libraries: 
        !pip install "ibm-watson-machine-learning>=1.0.327" | tail -n 1
        !pip install "pydantic>=1.10.0" | tail -n 1
        !pip install "langchain==0.0.340" | tail -n 1
    '''
    def __init__(self):
        self.credentials = {
            'url': IBM_ML_ENDPOINT_URL,
            'apikey': IBM_CLOUD_API_KEY
        }
        self.project_id = WATSONXAI_PROJECT_ID
    
    def list_models(self):
        print([model.name for model in ModelTypes])
    
    def set_model(self, model_id):
        self.model_id = ModelTypes[model_id]
    
    def list_decoding_methods(self):
        print([deco.name for deco in DecodingMethods])
    
    def set_decoding_modethod(self, decoding_method):
        self.decoding_method = DecodingMethods[decoding_method]
        self.describe_parameters()
    
    def describe_parameters(self):
        describe = '''
        Here you have the description of the Parameters: 
        Base on the watsonxai prompt lab we can see that this is how you need to setup
        the params depending if you use GREEDY or SAMPLE decoding method.

        When you use GREEDY decoding method:
        example = {
            "MAX_NEW_TOKENS": int, #int
            "MIN_NEW_TOKENS": int, #int
            "STOP_SEQUENCES": [], # list of stop seqences , is optional
            "REPETITION_PENALTY": 1, # 1.0 to 2.0
            "TEMPERATURE": 0,  #0 to 2,
        }
        - 

        When you use SAMPLE decoding method:
        example = {
            "MAX_NEW_TOKENS": int, #int
            "MIN_NEW_TOKENS": int, #int
            "STOP_SEQUENCES": [], # list of stop seqences , is optional
            "REPETITION_PENALTY": 1, # 1.0 to 2.0
            "RANDOM_SEED": None, # has to be None or an Integer
            "TEMPERATURE": float,  #0 to 2,
            "TOP_K": 50, #1 to 100 - default is 50. 
            "TOP_P": 1, #0 to 1 - default is 1.
        }
        
        '''
        print(describe)
    
    def set_model_params(self, params: dict):
        '''
        '''
        if self.decoding_method == DecodingMethods['GREEDY']:
            self.parameters = {
                GenParams.DECODING_METHOD: self.decoding_method,
                GenParams.MAX_NEW_TOKENS: params['MAX_NEW_TOKENS'],
                GenParams.MIN_NEW_TOKENS: params['MIN_NEW_TOKENS'],
                GenParams.STOP_SEQUENCES: params['STOP_SEQUENCES'],
                GenParams.REPETITION_PENALTY: params['REPETITION_PENALTY'],
                GenParams.TEMPERATURE: params['TEMPERATURE']
            }

        if self.decoding_method == DecodingMethods['SAMPLE']:
            self.parameters = {
                GenParams.DECODING_METHOD: self.decoding_method,
                GenParams.MAX_NEW_TOKENS: params['MAX_NEW_TOKENS'],
                GenParams.MIN_NEW_TOKENS: params['MIN_NEW_TOKENS'],
                GenParams.STOP_SEQUENCES: params['STOP_SEQUENCES'],
                GenParams.REPETITION_PENALTY: params['REPETITION_PENALTY'],
                GenParams.RANDOM_SEED: params['RANDOM_SEED'],
                GenParams.TEMPERATURE: params['TEMPERATURE'],
                GenParams.TOP_K: params['TOP_K'],
                GenParams.TOP_P: params['TOP_P']
            }  
    
    def create_base_llm_model(self):
        self.base_llm = Model(
            model_id=self.model_id,
            params=self.parameters,
            credentials=self.credentials,
            project_id=self.project_id
        )

    def create_lang_chain_llm(self):
        self.lang_chain_llm = self.base_llm.to_langchain()
        #self.lang_chain_llm.to_langchain()

        
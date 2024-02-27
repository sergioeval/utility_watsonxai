import requests 
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

# this is a comment from code spaces.

class WatsonxAiLangchain:
    '''
    To make available watsonxai usage 
    
    '''
    def __init__(self, 
                modelId, 
                projectId, 
                cloudApiKey, 
                stopSequences=[],
                repetitionPenalty= 1,
                decodingMethod='greedy',
                minTokens=0,
                maxTokens=50, 
                randomSeed=None,
                temperature=0,
                topK =50,
                topP = 1,):
        self.modelId = modelId
        self.projectId = projectId
        self.cloudApiKey = cloudApiKey
        self.decodingMethod = decodingMethod
        self.maxTokens = maxTokens
        self.minTokens = minTokens
        self.stopSequences = stopSequences
        self.repetitionPenalty = repetitionPenalty
        self.randomSeed = randomSeed
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
    
    def create_model(self):

        if self.decodingMethod == 'greedy':
            gen_params = {
                GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
                GenParams.MAX_NEW_TOKENS: self.maxTokens,
                GenParams.MIN_NEW_TOKENS: self.minTokens,
                GenParams.STOP_SEQUENCES: self.stopSequences,
                GenParams.REPETITION_PENALTY: self.repetitionPenalty
            }
           
        if self.decodingMethod == 'sample':

            gen_params = {
                GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
                GenParams.MAX_NEW_TOKENS: self.maxTokens,
                GenParams.MIN_NEW_TOKENS: self.minTokens,
                GenParams.STOP_SEQUENCES: self.stopSequences,
                GenParams.REPETITION_PENALTY: self.repetitionPenalty,
                GenParams.RANDOM_SEED: self.randomSeed,
                GenParams.TEMPERATURE: self.temperature,
                GenParams.TOP_K: self.topK,
                GenParams.TOP_P: self.topP
            }
        

        self.model = Model(
                model_id=self.modelId,
                params=gen_params,
                credentials={
                    "apikey": self.cloudApiKey,
                    "url": 'https://us-south.ml.cloud.ibm.com'
                },
                project_id=self.projectId
            )
        
        return self.model
    
    def langchain_llm(self):
        self.model_langchain = self.model.to_langchain()
        return self.model_langchain
    

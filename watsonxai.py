import requests 
from ibm_watson_machine_learning.foundation_models import Model

class WatsonxAi:
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
                maxTokens=50):
        self.modelId = modelId
        self.projectId = projectId
        self.cloudApiKey = cloudApiKey
        self.decodingMethod = decodingMethod
        self.maxTokens = maxTokens
        self.minTokens = minTokens
        self.stopSequences = stopSequences
        self.repetitionPenalty = repetitionPenalty
    
    def create_model(self):
        credentials = { 
            "url"    : "https://us-south.ml.cloud.ibm.com", 
            "apikey" : self.cloudApiKey
        }

        gen_parms = { 
            "DECODING_METHOD" : self.decodingMethod, 
            "MIN_NEW_TOKENS" : self.minTokens, 
            "MAX_NEW_TOKENS" : self.maxTokens,
            "STOP_SEQUENCES" : self.stopSequences,
            "REPETITION_PENALTY": self.repetitionPenalty
        }

        self.model = Model( self.modelId, credentials, gen_parms, self.projectId )
        return self.model

    def run_prompt(self, prompt):
        res = self.model.generate(prompt=prompt)
        return res
    

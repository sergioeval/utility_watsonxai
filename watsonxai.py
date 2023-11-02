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
        credentials = { 
            "url"    : "https://us-south.ml.cloud.ibm.com", 
            "apikey" : self.cloudApiKey
        }

        if self.decodingMethod == 'greedy':
            gen_parms = { 
                "DECODING_METHOD" : self.decodingMethod, 
                "MIN_NEW_TOKENS" : self.minTokens, 
                "MAX_NEW_TOKENS" : self.maxTokens,
                "STOP_SEQUENCES" : self.stopSequences,
                "REPETITION_PENALTY": self.repetitionPenalty
            }
        if self.decodingMethod == 'sample':
            gen_parms = { 
                "DECODING_METHOD" : self.decodingMethod, 
                "MIN_NEW_TOKENS" : self.minTokens, 
                "MAX_NEW_TOKENS" : self.maxTokens,
                "STOP_SEQUENCES" : self.stopSequences,
                "REPETITION_PENALTY": self.repetitionPenalty,
                "RANDOM_SEED": self.randomSeed,
                "TEMPERATURE": self.temperature,
                "TOP_K": self.topK,
                "TOP_P": self.topP
            }
            
        self.model = Model( self.modelId, credentials, gen_parms, self.projectId )
        
        return self.model

    def run_prompt(self, prompt):
        res = self.model.generate(prompt=prompt)
        return res
    

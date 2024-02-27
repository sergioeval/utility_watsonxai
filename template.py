#%% importing the class 
# using python 12 langchain

from watsonxai_langchain_v2 import WatsonxAiLangchain

#%% create object 
llm = WatsonxAiLangchain()

#%%list models 

llm.list_models()


# %% Set model 
llm.set_model(model_id='FLAN_UL2')

# %% list decodng methods 
llm.list_decoding_methods()

# %% set decoding methods
llm.set_decoding_modethod(decoding_method='GREEDY')

# %% set up params

llm.set_model_params(params={
            "MAX_NEW_TOKENS": 200,#int
            "MIN_NEW_TOKENS": 0, #int
            "STOP_SEQUENCES": [], #list of stop seqences , is optional
            "REPETITION_PENALTY": 1 #1.0 to 2.0
        })

    

# %%Create base llm model

llm.create_base_llm_model()

# %% create langchain llm

llm.create_lang_chain_llm()



# %%

llm.lang_chain_llm(prompt='Talking about physics who can be the best scientific of all times?')
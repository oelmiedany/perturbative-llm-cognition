from perturbative_llm_cognition import core

model_id = 'mistralai/Mistral-7B-Instruct-v0.3'
tokenizer, model = core.load_tokenizer_and_model(model_id)

core.conversation_loop(tokenizer, model)
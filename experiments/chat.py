from perturbative_llm_cognition import core

tokenizer, model = core.load_tokenizer_and_model()

core.conversation_loop(tokenizer, model, max_new_tokens=258)
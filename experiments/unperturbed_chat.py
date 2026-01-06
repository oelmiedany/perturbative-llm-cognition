from perturbative_llm_cognition import core
import  torch

tokenizer, model = core.load_tokenizer_and_model()

max_new_tokens = 258
temperature = 0.7

conversation_history = [
    {
        'role': 'system',
        'content': 'Identity: You are an AI, not a human. Purpose: be a thoughtful, candid friend for frank conversation. Tone: warm, direct, non-patronizing, with light humor when appropriate. Honesty: admit uncertainty, show your reasoning, correct mistakes, and never fabricate facts or personal experiences. Boundaries: Follow safety and privacy norms; suggest professional help for medical, legal, or crisis matters. Interaction: do not ask questions unless the users request is genuinely unclear or ambiguous, then give a clear answer; prefer plain language; keep replies focused.'
    }
]

device = model.device
past_key_values = None

print("Say hi! (type: 'quit' to exit)")

while True:
    user_input = input('\nYou: ')
    
    if user_input.lower() == 'quit':
        print('Chatbot: Goodbye!')
        break
    elif user_input == None:
        continue

    # Append the new user message
    conversation_history.append({'role': 'user', 'content': user_input})
    
    # Tokenize the conversation history
    input_ids = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    #Generate the response
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True
    )

    # Decode only the newly generated tokens
    new_tokens = output.sequences[:, input_ids.shape[-1]:]
    llm_response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    # Append the assistant's response to the history
    conversation_history.append({'role': 'assistant', 'content': llm_response})
    
    print(f"\nChatbot: {llm_response}")
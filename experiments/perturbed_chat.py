from perturbative_llm_cognition import lsd
import  torch

model = lsd.LSDPerturbedLLM(
    layer_start=5,
    layer_end=25,
    attention_scaling_factor=1.7,
    attention_noise=0.3,
    attention_diagonal_penalty=-0.9,
    attention_probability_smoothing_factor=0.5,
    js_tolerance=0.3,
    perturbation_blend_min=0.1,
    perturbation_blend_max=0.3,
)



max_new_tokens = 258

print("\nPerturbative LLM Cognition: An exploration of the 'thinking' metaphor in LLM chatbots")
print("Say hi to your friend, but careful he might be on something... (type: 'quit' to exit)")

while True:
    user_input = input('\nYou: ')
    
    if user_input.lower() == 'quit':
        print('Chatbot: Goodbye!')
        break
    elif user_input == None:
        continue
    
    llm_response = model.generate_with_leash(
        prompt= user_input,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.9,
        only_new_tokens=True
    )
    
    print(f"\nChatbot: {llm_response}")
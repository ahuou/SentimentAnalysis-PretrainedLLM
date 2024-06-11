 #connect to the grid:
   # qrsh -l q_short_gpu -P PROJECTNAME -l hostname=vgni* -now no
# Create conda environment "llm"
# conda activate llm
# Install llama.cpp-python with CUDA:
   # CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
# Now you can run the example:
   # python example.py

from data_retriever import data1, data2
from llama_cpp import Llama
import pickle
import os
import random
def get_mult_instructions(file):
   contexts = []
   prompts = []

   #Set a flag to distinguish between contexts and prompts
   reading_contexts = True  # Start by reading contexts

   # Open the prompts_example.txt file and read line by line
   with open(file, 'r') as file:
      for line in file:
         # Strip the line of leading/trailing whitespace
         clean_line = line.strip()
         # Skip empty lines
         if not clean_line:
            # An empty line signifies the end of contexts and the beginning of prompts
            reading_contexts = False
            continue
            # Append the line to the appropriate list based on the current reading flag
         if reading_contexts:
            contexts.append(clean_line)
         else:
            prompts.append(clean_line)
   return contexts, prompts

def get_single_instructions(file):
   context = ""
   prompt = ""

   # Open the file and read the lines
   with open(file, 'r') as file:
      # Read all lines into a list
      lines = file.readlines()

      # Extract the first line into context
      if lines:
         context = lines[0].strip()

      # Extract the rest of the lines into prompt
      if len(lines) > 1:
         prompt = ''.join(lines[1:]).strip()

   # Print the variables (for demonstration purposes)
      print("Context:", context)
      print("Prompt:", prompt)
   return context, prompt


PATH_MODEL_LLAMA13B = "llama-2-13b-chat.Q8_0.gguf"
PATH_MODEL_MISTRAL7B = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
PATH_MODEL = "llama-2-7b.Q6_K.gguf"
PARAMS_GENERATION = {"max_tokens": 300, "stop": ["\n"], "temperature": 0, "echo": False}
PARAMS_MODEL = {"n_ctx": 10240, "n_batch": 300, "n_threads": None, "n_gpu_layers": -1, "seed": 13}

# special tokens used by llama 2 chat
B_SEQ, E_SEQ = "<s>", "</s>"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


conversation_example = """
Agent: Thank you for calling , how may I help you?
Customer: Hi there. Um, I'm calling, uh, about a ticket, uh, the 
issue that we have h- bringing our chargers online to the, to the server.
Agent: Sure. I will help you with that. Could you please provide me the ticket number?
Customer: Yes. The ticket number is zero zero three six zero seven four four.
Agent: Please wait on the call while I pull up this date of for you. Please be on line, ma'am.
Customer: Okay.
Agent: So may I know your first name, so am I speaking to?
Customer: Uh, yes. This is Jewel Williams, J E W E L.
Agent: Okay, Jewel. So hold on for me a moment, let me check this.
Customer: Okay.
Agent: Yeah, currently, uh, I could see the latest updates on this case is... Uh, uh, our technical team is working on this ticket, so you can expect a call back from our technical team shortly to proceed with further troubleshooting steps, okay?
Customer: O- oh, okay. Um, someone-
Customer: ... someone did try to call me back, um, but I wasn't- ... on site.
Agent: So-
Agent: Okay.
Agent: Okay. All right. Could you please provide me your available 
time? So when would be... Uh, you available at this site?
Customer: Um, so I, I am only available at site today. I will be leaving shortly, and then I will not be at site again for some time. I might have someone that can assist from site, but this was, you know, to put... We worked on this all day yesterday and today, and it still is not working.
Agent: Okay. Okay. I'll ask, uh, ask someone from technical team to reach out to you right away, okay?
Customer: Okay, thank you.
Agent: Thank you so much, Jewel.
Agent: We will-... contact you as soon as possible. Thanks for calling Siemens Mobility Team. Have a great day.
Customer: Bye.
Customer: Okay, thank you.
"""


def llama2instruct(msg, task_context=None, context=None):
   context = context or task_context
   inst = B_SEQ + B_INST + B_SYS + context + E_SYS + msg + E_INST
   #print("Instr", inst)
   return inst

def prompter(textdf, randomize_few_shot=False, marker="DS1"):
   if not randomize_few_shot:
      return
   else:
      if marker == "DS1":
         maskPos = text_df["Sentiment"].isin(["Positive"])
         maskNeg = text_df["Sentiment"].isin(["Negative"])
         posTexts = text_df[maskPos]
         negTexts = text_df[maskNeg]
         return posTexts, negTexts
      else:
         maskPos = text_df2["Sentiment"].isin(["Positive"])
         maskNeutral = text_df2["Sentiment"].isin(["neutral"])
         maskNeg = text_df2["Sentiment"].isin(["Negative"])

         posTweets = text_df2[maskPos]
         neutralTweets = text_df2[maskNeutral]
         negTweets = text_df2[maskNeg]
         return posTweets, neutralTweets, negTweets


# batch size is not usable right now as the model isn't batch trained yet, but I will leave this feature in case it becomes available in the future
def run_model(prompts_file, text_df, path_model, params_model, params_generation, batch_size, marker, new_marker="", start=0, random_few_shot=False, classified=False):

   emotions = []

   #contexts, prompts = get_mult_instructions(prompts_file)

   context, prompt = get_single_instructions(prompts_file)

   model = Llama(path_model, **params_model)

   model.reset()


   #Iterate over the actual number of different prompts

   TASK_CONTEXT = context
   INSTRUCTION = prompt
   emotion_prompt = []
   print(context, prompt)


   #Iterate over the needed number of loops to go through full text (changes if batches or no batches)
   if(batch_size > 0):
      u = int(text_df.shape[0]/batch_size)
      print("Num_batches: ", u)
   else:
      u = text_df.shape[0]
      batch_size = 1
   for j in range(start, u):

      k = batch_size*j
      next_k = batch_size*(j+1)

      print("Running batch", next_k)

      #print("Texto: ", str(text_df.iloc[(k):next_k, 0]))
      #prompt_example = llama2instruct("Please analyze the following text and provide a sentiment score between 0 (negative) and 1 (positive):"
 #+ str(text_df.iloc[(k): next_k, 0]+"\n Score: ")) + "\nIntent label: \""

      defPrompt = context + prompt
      if random_few_shot:
         if marker == "DS1":
            mode = "Text: "
            if(classified):
               ran = random.randint(0, posTexts["Text"].size)
               defPrompt = defPrompt + "Text: " + str(posTexts.iloc[ran, 0]) +"\n Sentiment:  " + str(posTexts.iloc[ran, 1]) + "\n\n"
               ran = random.randint(0, negTexts["Text"].size)
               defPrompt = defPrompt + "Text: " + str(negTexts.iloc[ran, 0]) + "\n Sentiment: " + str(negTexts.iloc[ran, 1])
               print(defPrompt)
            else:
               for _ in range(2):
                  ran = random.randint(0, text_df["Text"].size)
                  defPrompt = defPrompt + "Text: " + str(text_df.iloc[ran, 0]) + "\n Sentiment:  " + str(text_df.iloc[ran, 1]) + "\n\n"


         else:
            mode = "Tweet: "
            if(classified):
               ran = random.randint(0, text_df["Text"].size) - 1
               defPrompt = defPrompt + "Tweet: " + str(text_df.iloc[ran, 0]) + "\n Sentiment: " + str(text_df.iloc[ran, 1]) + "\n\n"
               ran = random.randint(0, text_df["Text"].size)
               defPrompt = defPrompt + "Tweet: " + str(text_df.iloc[ran, 0]) + "\n Sentiment: " + str(text_df.iloc[ran, 1]) + "\n\n"
               ran = random.randint(0, text_df["Text"].size)
               defPrompt = defPrompt + "Tweet: " + str(text_df.iloc[ran, 0]) + "\n Sentiment: " + str(text_df.iloc[ran, 1])
               print(defPrompt)
            else:
               for _ in range(3):
                  ran = random.randint(0, text_df["Text"].size) - 1
                  defPrompt = defPrompt + "Tweet: " + str(text_df.iloc[ran, 0]) + "\n Sentiment:  " + str(text_df.iloc[ran, 1]) + "\n\n"


      prompt_example = defPrompt + mode + str(text_df.iloc[k:(next_k-1), 0]) + "\n Sentiment: "
      print(prompt_example)

      intent = model(prompt_example, **params_generation)
      print("Intent: ", intent)
      intent_choices = intent["choices"]
      intent_first_choice = intent_choices[0]["text"]
      #intent_first_choice_text = intent_first_choice[:4]
      intent_first_choice_text = intent_first_choice
      print(f"\n\nPredicted intent: \"{intent_first_choice_text}\"")
      emotion_prompt.append(intent_first_choice_text)
      print("List quickpeak: ", emotion_prompt[-3:])
      pathmini = "res"+str(j)+new_marker+".txt"
      file = open(pathmini, 'wb')
      pickle.dump(emotion_prompt, file)
      file.flush()
      file.close()
      if(j%10 != 0 and j != u-1):
         os.remove(pathmini)
      elif(j != 0 and j%100 == 0):
         for ind in range(1, 10):
            pathmin = "res"+str(j-10*ind)+".txt"
            os.remove(pathmin)
         if (j%1000 == 0):
            for ind in range(1, 10):
               lilpath = "res"+str(j-100*ind)+".txt"
               os.remove(lilpath)
      emotions.append(emotion_prompt)
      path = "full_res"+marker+new_marker+".txt"
      file2 = open(path, 'wb')
      pickle.dump(emotions, file2)
      file2.flush()
      file2.close()



 # Batches with 10 different prompts with examples when learning prompt engineering
#run_model('prompts_example.txt', conversation_example, 10, PATH_MODEL, PARAMS_MODEL, PARAMS_GENERATION, 0)


text_df = data1
text_df2 = data2
print(text_df.shape[0])
posTexts, negTexts = prompter(text_df, randomize_few_shot=True, marker="DS1")
print(posTexts)
posTweets, neutralTweets, negTweets = prompter(text_df2, randomize_few_shot=True, marker="DS2")



#run_model('prompts.txt', text, 3, PATH_MODEL, PARAMS_MODEL, PARAMS_GENERATION)

run_model('prompts/DS2_3_Shot_prompt.txt', text_df2, PATH_MODEL_MISTRAL7B, PARAMS_MODEL, PARAMS_GENERATION, 0, marker='DS2', start=14270, random_few_shot=True, classified=False)
#run_model('newPromptsTwitter.txt', text_df2, PATH_MODEL_MISTRAL7B, PARAMS_MODEL, PARAMS_GENERATION, 0,  marker='DS2')




















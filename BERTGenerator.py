import pandas as pd
import numpy as np
import os
import re
import concurrent.futures
from transformers import BertTokenizer, BertForMaskedLM
import torch

from transformers import DistilBertTokenizer, DistilBertForMaskedLM


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Load the pre-trained DistilBERT model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

model = DistilBertForMaskedLM.from_pretrained(model_name)
model.eval()


# Different BERT model

# model_name = 'bert-base-uncased'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForMaskedLM.from_pretrained(model_name)
# model.eval()

# Check if a GPU is available, and if so, move the model to the GPU
if torch.cuda.is_available():
    model = model.cuda()



# Load and preprocess data
# dir = '~/DataAnalysis/data/Sprints/HighRes/'
df = pd.read_hdf('../data/WindyMASigned.h5')
df1=df[15000:18000]
df1=df1.round(3)

# Test Dataset
test=df[15000:15100]
test=test.round(3)


# Set the chunk size
chunk_size = 60

# Divide the DataFrame into smaller chunks
num_chunks = (len(df1) - 1) // chunk_size + 1

for chunk_index in range(num_chunks):
    start_index = chunk_index * chunk_size
    end_index = min((chunk_index + 1) * chunk_size, len(df1))
    chunk_df = df1.iloc[start_index:end_index]

    # Convert the chunk DataFrame to textual format
    text_data = ""
    for idx, row in chunk_df.iterrows():
        text_data += f"Odor encounter: {row['odor']}, Location: ({row['xsrc']}, {row['ysrc']}), U velocity: {row['U']}, V velocity: {row['V']}.\n"

print('Done Chunking process')


def extract_values(generated_text):
    pattern = r"odor encounter\s*:\s*([\d.]*)\s*([\d.]*)\s*,\s*location\s*:\s*\(\s*([\d.]*)\s*([\d.]*)\s*,\s*([\d.]*)\s*([\d.]*)\s*\)"
    match = re.search(pattern, generated_text)
    
    if match:
        odor = float(match.group(1) + match.group(2))
        xsrc = float(match.group(3) + match.group(4))
        ysrc = float(match.group(5) + match.group(6))
        

        return {"odor": odor, "xsrc": xsrc, "ysrc": ysrc}
    else:
        return None

def generate_odor_with_bert(input_row):
    prompt = f"{text_data}\nGenerate a new odor encounter with the following location: ({input_row['xsrc']}, {input_row['ysrc']}) and U velocity: {input_row['U']}, V velocity: {input_row['V']}.Please use the format: Odor encounter: X, Location: (X, Y)"
    
    # print(prompt)
    match = None
    max_iterations = 50
    iteration = 0
    
    while not match and iteration < max_iterations:
        
        # Tokenize the input prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True)
     
        mask_idx = len(input_ids[0]) - 1

        # Mask the last token (X)
        input_ids[0, mask_idx] = tokenizer.mask_token_id

        # Move input tensor to GPU if available
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        # Generate logits for masked token
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Sample the top 10 predicted tokens
        top_k = 10
        probs = torch.nn.functional.softmax(logits[0, mask_idx], dim=-1)
        top_k_indices = torch.topk(probs, top_k).indices

        # Generate text for each predicted token
        for index in top_k_indices:
            generated_text = tokenizer.decode(input_ids[0].tolist()[:-1] + [index.item()]).strip()
            match = extract_values(generated_text)
        
            if match:
                break
                
        iteration +=1
    
    if match:

        return match
    else:
        print("Failed to parse generated text")
        print("Generated text:", generated_text)
        return None


def main():

    # Create a new DataFrame to store generated data
    generated_df = pd.DataFrame(columns=['odor', 'xsrc', 'ysrc'])
    num_threads = 14 
    print('Generating Data')
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(generate_odor_with_bert, test.to_dict('records')))
 
    print('Done Generating Data')
    for result in results:
        if result is not None:
            generated_df = pd.concat([generated_df, pd.DataFrame([result])], ignore_index=True)

    generated_df = generated_df.astype(float)
    generated_df.to_hdf('../generated.h5',key='generated_df',mode='w')



if __name__ == "__main__":
    main()

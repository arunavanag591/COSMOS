import openai
import pandas as pd
import re
import openpyxl
import concurrent.futures



# Load and preprocess data
dir = '~/DataAnalysis/data/Sprints/HighRes/'
df = pd.read_hdf(dir+'Windy/WindyMASigned.h5')
df1=df[0:150000]
df1=df1.round(3)

# Test Dataset
test=df[15000:25000]
test=test.round(3)




## Get Key
def read_first_cell(file_path):
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active
    first_cell = sheet.cell(row=1, column=1)
    return first_cell.value

file_path = '../openaikey.xlsx'  
first_cell_value = read_first_cell(file_path)

openai.api_key=first_cell_value
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

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



def generate_odor(input_row):
    prompt = f"{text_data}\nGenerate a new odor encounter with the following location: ({input_row['xsrc']}, {input_row['ysrc']}) and U velocity: {input_row['U']}, V velocity: {input_row['V']}.Please use the format: Odor encounter: X, Location: (X, Y)"
    pattern = r"Odor encounter: ([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?), Location: \(([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?), ([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\)"

    # pattern = r"Odor encounter: (.*), Location: \((.*), (.*)\)"
    match = None

    while not match:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=0.7,
            max_tokens=50,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        generated_texts = response.choices[0].text.strip()
        match = re.search(pattern, generated_texts)
        if not match:
            print("Failed to parse generated text")
            print("Generated text:", generated_texts)

    if match:
        new_entry = {
            "odor": float(match.group(1).strip()),
            "xsrc": float(match.group(2).strip()),
            "ysrc": float(match.group(3).strip()),
        }
        return new_entry
    else:
        print("Failed to parse generated text")
        print("Generated text:", generated_texts)
        return None
    



def main():
 
    # Create a new DataFrame to store generated data
    generated_df = pd.DataFrame(columns=['odor', 'xsrc', 'ysrc'])
    num_threads = 8 

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(generate_odor, test.to_dict('records')))

    for result in results:
        if result is not None:
            generated_df = pd.concat([generated_df, pd.DataFrame([result])], ignore_index=True)

    generated_df.to_hdf('../generated.h5',key='generated_df',mode='w')



if __name__ == "__main__":
    main()
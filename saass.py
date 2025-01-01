
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT model for Turkish from Stefan IT repository
model_name = 'dbmdz/bert-base-turkish-128k-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

# Step 1: Load Data
answer_sheet = pd.read_excel('answer_sheet.xlsx')
student_answers = pd.read_excel('answers.xlsx')

# Prepare output dataframe
output_columns = ['Student ID', 'Student Total Score']
for i in range(1, 5):
    output_columns.extend([
        f'Q{i}-Predicted Score', f'Q{i}-Matching Answer', f'Q{i}-Cosine Similarity'
    ])
output_data = []

# Step 2: Calculate Scores
for index, student in student_answers.iterrows():
    student_id = student['Unnamed: 0']
    total_score = 0
    student_row = [student_id]
    
    for q_num in range(1, 5):
        question_id = f'Q{q_num}'
        student_answer = student[f'{question_id}-answer']
        if pd.isnull(student_answer):
            student_row.extend([0, None, None])
            continue
        
        best_score = 0
        best_answer = None
        best_similarity = 0
        
        for score_set in range(1, 4):
            answers = answer_sheet.loc[answer_sheet['Question- ID'] == question_id, f'SCORE-{score_set}'].values[0]
            answers_list = [a.strip() for a in answers.split(',')]
            
            for exp_answer in answers_list:
                student_embedding = get_embedding(student_answer)
                expected_embedding = get_embedding(exp_answer)
                similarity = cosine_similarity(student_embedding, expected_embedding)[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_answer = exp_answer
                    best_score = score_set
        
        total_score += best_score
        student_row.extend([best_score, best_answer, best_similarity])
    
    student_row.insert(1, total_score)
    output_data.append(student_row)

# Step 3: Save Results
output_df = pd.DataFrame(output_data, columns=output_columns)
output_df.to_excel('output.xlsx', index=False)

print("Scoring completed. Results saved to output.xlsx")

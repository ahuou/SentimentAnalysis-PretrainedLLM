import pandas as pd
import os
import sqlite3

def feelings_to_emotions(emotion_list, neg_list, pos_list):
    for i in range(len(emotion_list)):
        if emotion_list[i] in neg_list:
            emotion_list[i] = 'Negative'
        elif emotion_list[i] in pos_list:
            emotion_list[i] = 'Positive'


def safe_to_csv(df, filename):
    if not os.path.exists(filename):
        df.to_csv(filename, index=False, sep='\t')
    else:
        print(f"File {filename} already exists. Not overwriting.")

# Example usage


text = []
labs = []
emotion = []
label_map = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]   # information provided with the dataset
with open('dataset_textEmotion/text.txt', 'r') as f:
     for line in f:
        line = line.strip()
        label = line[1:line.find(']')].strip().split()
        sent = line[line.find(']')+1:].strip()
        text.append(sent)
        labs.append(label)

for i in range(len(labs)):
    for j in range(len(labs[0])):
        if labs[i][j]=='1.':
            emotion.append(label_map[j])

raw_data1 = pd.concat((pd.Series(text), pd.Series(emotion)), axis=1, ignore_index=True)
raw_data1.columns = ['Text','Emotion']
feelings_to_emotions(emotion, ["fear", "anger", "sadness", "disgust", "shame", "guilt"], ['joy'])
data1 = pd.concat((pd.Series(text),pd.Series(emotion)), axis=1, ignore_index=True)
data1.columns = ['Text','Sentiment']



raw_data2 = pd.read_csv('dataset_emoDetectInText/EmotionDetectInText/data/emotion_dataset.csv')
raw_data2 = raw_data2[['Clean_Text', 'Emotion']]
raw_data2.columns = ['Text', 'Emotion']
mask = raw_data2['Emotion'] == 'surprise'
raw_data2 = raw_data2[~mask]

emotion2 = raw_data2['Emotion'].values
text2 = raw_data2['Text'].values
new_emotion2 = emotion2.copy()
feelings_to_emotions(new_emotion2, ['disgust', 'sadness', 'fear', 'anger', 'shame'], ['joy'])


data2 = pd.concat((pd.Series(text2),pd.Series(new_emotion2)), axis=1, ignore_index=True)
data2.columns = ['Text', 'Sentiment']

safe_to_csv(raw_data1, 'simple_emotions.txt')
#raw_data1.to_csv('simple_emotions.txt', index=False, sep='\t')
safe_to_csv(data1, 'simple_sentiments.txt')
#data1.to_csv('simple_sentiments.txt', index=False, sep='\t')
safe_to_csv(raw_data2, 'clean_tweet_emotions.txt')
#raw_data2.to_csv('clean_tweet_emotions.txt', index=False, sep='\t')
safe_to_csv(data2, 'clean_tweet_sentiments.txt')
#data2.to_csv('clean_tweet_sentiments.txt', index=False , sep='\t')

print(data1.head())
print(data1.columns)
print(data1)
from data_retriever import raw_data1, raw_data2, data1, data2

import nltk
from nltk.corpus import stopwords
import string
from collections import Counter

# Download the set of stop words the first time
nltk.download('stopwords')

# Load stop words
stop_words = set(stopwords.words('english'))
stop_words.update(['Im', "I'm", "im", "i'm"])


def clean_and_tokenize(text):
    if not isinstance(text, str):
        text = str(text)  # Convert to string if it's not already
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    return [word for word in text.split() if (word not in stop_words)]

def save_to_file(filename, stat_var, stat_var_names, most_common_words, common_words_per_emotion):
    # Open the file with write mode, which will create it if it doesn't exist
    with open(filename, "w") as file:
        # Redirecting print statements to the file

        for i in range(len(stat_var)):
            print(f" {stat_var_names[i]}: {stat_var[i]}", file=file)

        # Printing most_common_words
        print("\n most_common_words:", file=file)
        for word, count in most_common_words:
            print(f"  {word}: {count}", file=file)

        # Printing common_words_per_emotion
        print("\n common_words_per_emotion:", file=file)
        for emotion, common_words in common_words_per_emotion.items():
            print(f"\n {emotion}: ", file=file)
            for word, count in common_words:
                print(f" \t   {word}: {count}", file=file)


def var_maker(data_type, data_name, data_type_name, out_file_name):
    num_instances1 = data_type.shape[0]
    average_length1 = data_type['Text'].str.len().mean()
    words1 = data_type['Text'].str.split(expand=True).stack()
    vocabulary_size1 = words1.unique().size
    number_of_tokens1 = words1.size
    emotion_distribution1 = data_type[data_type_name].value_counts()
    length_std1 = data_type['Text'].str.len().std()
    length_var1 = data_type['Text'].str.len().var()
    new_data_type = data_type.copy()
    new_data_type['Text'] = new_data_type['Text'].apply(clean_and_tokenize)
    flat_words1 = [word for sublist in new_data_type['Text'] for word in sublist]
    counter1 = Counter(flat_words1)
    most_common_words1 = counter1.most_common(10)
    # Word frequency by emotion

    common_words_per_emotion1 = new_data_type.groupby(data_type_name)['Text'].apply(
        lambda texts: Counter(
            word for word in " ".join(str(x) for x in texts).split() if
            word.lower() not in stop_words and word not in string.punctuation
        ).most_common(5)
    )
    stat_var1 = [num_instances1, average_length1, vocabulary_size1, number_of_tokens1, emotion_distribution1,
                 length_std1, length_var1]
    stat_var_names1 = [f"num_instances_{data_name}", f"average_length_{data_name}", f"vocabulary_size_{data_name}", f"number_of_tokens_{data_name}",
                       f"emotion_distribution_{data_name}", f"length_std_{data_name}", f"length_var_{data_name}"]

    save_to_file(out_file_name, stat_var1, stat_var_names1, most_common_words1, common_words_per_emotion1)



var_maker(data1, 'simple_sentiment', 'Sentiment',"dataset_simple_sentiments.txt")
var_maker(raw_data1, 'simple_emotion', 'Emotion', "dataset_simple_emotions.txt")
var_maker(data2, 'tweet_sentiment',  'Sentiment',"dataset_tweets_sentiments.txt")
var_maker(raw_data2, 'tweet_emotion', 'Emotion', "dataset_tweets_emotions.txt")

print(data1)
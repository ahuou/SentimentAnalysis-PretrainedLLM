import pickle
from data_retriever import data1, data2

text_df = data1
text_df2 = data2
print(text_df.shape[0])
maskPos = [text_df["Sentiment"].isin(["Positive"]), text_df2["Sentiment"].isin(["Positive"])]
maskNeg = [text_df["Sentiment"].isin(["Negative"]), text_df2["Sentiment"].isin(["Negative"])]
maskNeutral = [text_df2["Sentiment"].isin(["neutral"])]
posTexts = text_df[maskPos[0]]
negTexts = text_df[maskNeg[0]]
posTweets = text_df2[maskPos[1]]
neutralTweets = text_df2[maskNeutral[0]]
negTweets = text_df2[maskNeg[1]]

print("PosText: ", posTexts[:5], "NegText: ", negTexts[:5])
print("PosTweet: ", posTweets[:5], "neutralTweet: ", neutralTweets[:5], "negTweet: ", negTweets[:5])
print("Lolita", posTexts.iloc[7, 1])
print("NEUTRAL: ", neutralTweets[:10])


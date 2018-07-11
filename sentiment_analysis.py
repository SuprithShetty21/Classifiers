import tweepy
from textblob import TextBlob
import csv

f=open('database.csv','w')
main_list = [ ]
# setting variables access related keys and secrets
consumer_key = 'NIX4Duq1L3dtpGsl7FmnJFqCj'
consumer_secret = '5juAzuNYYG7KBSAII8U7GcyjjyhbOfnkKjanAmfmCodrR0mljk'

access_token = '2835145946-EdIEwIAbnUgMA7mOUXiGl3atCOBCQBDKYf57pKG'
access_token_secret = 'LjQXC0w6x6qky6ZIuSIIkKQ30Fj3M5eDy1jAhSrSQER0m'

auth=tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# makes a list of tweets
public_tweets = api.search('Narendra Modi')


# function to return whether sentiments are positive or negative
def get_analysis(var,threshold = 0.0):
    if var >= threshold:
        return 'positive'
    else:
        return 'negative'


for tweet in public_tweets :
    list_ = [ ] 
    print(tweet.text)
    list_.append(tweet.text)

    analysis =TextBlob(tweet.text)   #performing sentiment analysis

    #var = analysis.sentences
    #list_.append(var)
    print(analysis.sentiment)
    print(' ')

    list_.append(get_analysis(var = analysis.sentiment.polarity))
    list_.append(get_analysis(var = analysis.sentiment.subjectivity,threshold = 0.5))

    main_list.append(list_)

print(main_list)

#with f:
   # writer = csv.writer(f)
    #writer.writerows(main_list)




import tweepy
from textblob import TextBlob
import csv

f=open('database.csv','w')
main_list = [ ]
row0 = ['Tweet','polarity','subjectivity']

# setting variables access related keys and secrets
# enter the key and secret token below
consumer_key= 'CONSUMER_KEY_HERE'
consumer_secret= 'CONSUMER_SECRET_HERE'

access_token='ACCESS_TOKEN_HERE'
access_token_secret='ACCESS_TOKEN_SECRET_HERE'

auth=tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# makes a list of tweets
public_tweets = api.search('Trump')


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

write_ = csv.writer(f)
write_.writerow(row0)
write_.writerows(main_list)
f.close()



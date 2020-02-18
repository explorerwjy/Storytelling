import csv
import pandas as pd
import socket
import sys
import requests
import requests_oauthlib
import json
import tweepy
import time

# Connecting to Twitter
my_auth = requests_oauthlib.OAuth1(CONSUMER_KEY, CONSUMER_SECRET,ACCESS_TOKEN, ACCESS_SECRET)



def ReadJson(json_obj):
    _id = json_obj["id"]
    _text = json_obj["text"]
    _created_at = json_obj["created_at"]
    _retweet_count = json_obj["retweet_count"]
    _user_id = json_obj["user"]["id"]
    _user_followers_count = json_obj["user"]["followers_count"]

    try:
        _rt_id = json_obj["retweeted_status"]["id"]
        _rt_text = json_obj["retweeted_status"]["text"]
        _rt_created_at = json_obj["retweeted_status"]["created_at"]
        _rt_retweet_count = json_obj["retweeted_status"]["retweet_count"]
        _rt_user_id = json_obj["retweeted_status"]["user"]["id"]
        _rt_user_followers_count = json_obj["retweeted_status"]["user"]["followers_count"]
    except:
        _rt_id = ""
        _rt_text = ""
        _rt_created_at = ""
        _rt_retweet_count = ""
        _rt_user_id = ""
        _rt_user_followers_count = ""

    return [_id, _text, _created_at, _retweet_count, _user_id, _user_followers_count, _rt_id, _rt_text, _rt_created_at, _rt_retweet_count, _rt_user_id, _rt_user_followers_count]

def Tweets2CSV(tweets, writer):
    for tweet in tweets:
        res = ReadJson(tweet._json)
        writer.writerow(res)

def RetriveDataset(tids, api, outname):
    writer = csv.writer(open("dat/%s"%outname, "wt"))
    header = ["id", "text", "created_at", "retweet_count", "user_id", "user_followers_count",
              "rt_id", "rt_text", "rt_created_at", "rt_retweet_count", "rt_user_id", "rt_user_followers_count"]
    writer.writerow(header)
    i = 0; j = 100
    while 1:
        try:
            tweet = api.statuses_lookup(tids[i:j])
        except RateLimitError:
            time.sleep(10)
            continue
        Tweets2CSV(tweet, writer)
        i += 100;
        j += 100;
        if j > len(tids):
            break
        if j >= len(tids):
            j = tids









#  call the Twitter API URL and return the response for a stream of tweets
def get_tweets():
    url = 'https://stream.twitter.com/1.1/statuses/filter.json'
    query_data = [('language', 'en'), ('locations', '-130,-20,100,50'),('track','#')]
    query_url = url + '?' + '&'.join([str(t[0]) + '=' + str(t[1]) for t in query_data])
    response = requests.get(query_url, auth=my_auth, stream=True)
    print(query_url, response)
    return response

#  takes the response from the above one and extracts the tweets’ text from the whole tweets’ JSON object
#  through TCP connection
def send_tweets_to_spark(http_resp, tcp_connection):
    fout = open("tweets.txt", 'wt')
    for line in http_resp.iter_lines():
        try:
            full_tweet = json.loads(line)
            tweet_text = full_tweet['text']
            #print("Tweet Text: " + tweet_text)
            fout.write(tweet_text+"\n")
        except:
            e = sys.exc_info()[0]
            print("Error: %s\t" % (e, sys.exc_info()[2]))

#TCP_IP = "0.0.0.0"
#TCP_PORT = 10002
#conn = None
#s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s.bind((TCP_IP, TCP_PORT))
#s.listen(1)
#print("Waiting for TCP connection...")
#conn, addr = s.accept()
#print("Connected... Starting getting tweets.")
#resp = get_tweets()
#send_tweets_to_spark(resp, conn)


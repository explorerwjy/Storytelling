import socket
import sys
import requests
import requests_oauthlib
import json

# Connecting to Twitter
my_auth = requests_oauthlib.OAuth1(CONSUMER_KEY, CONSUMER_SECRET,ACCESS_TOKEN, ACCESS_SECRET)


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

TCP_IP = "0.0.0.0"
TCP_PORT = 10002
#conn = None
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)
print("Waiting for TCP connection...")
conn, addr = s.accept()
print("Connected... Starting getting tweets.")
resp = get_tweets()
send_tweets_to_spark(resp, conn)


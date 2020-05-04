from key import *
import csv
import pandas as pd
import numpy as np
import socket
import sys
import requests
import requests_oauthlib
import json
import tweepy
import time
import pickle
import os
import argparse
import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from itertools import groupby
import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from datetime import datetime 
import networkx as nx
#import datetime

######################################################
# Get Data From Twitter
######################################################

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
    writer = csv.writer(open("../dat/%s"%outname, "wt"))
    header = ["id", "text", "created_at", "retweet_count", "user_id", "user_followers_count",
              "rt_id", "rt_text", "rt_created_at", "rt_retweet_count", "rt_user_id", "rt_user_followers_count"]
    writer.writerow(header)
    i = 0; j = 100
    while 1:
        try:
            tweet = api.statuses_lookup(tids[i:j])
        except:
            time.sleep(10)
            continue
        Tweets2CSV(tweet, writer)
        i += 100;
        j += 100;
        if j > len(tids):
            break
        if j >= len(tids):
            j = tids

######################################################
# NLP
######################################################

class TwitterCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.wnl = WordNetLemmatizer()
        self.stemmer = LancasterStemmer()
        self.urlsearch = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    def Clean(self, text):
        #text = re.sub(r'RT\s@[\w]*:', '', text)
        #text = re.sub(self.urlsearch, '', text)
        #text = re.sub(r'[^\w\s]', ' ', text)
        words = nltk.word_tokenize(text)
        new_words = []
        for word in words:
            if word.isnumeric():
                continue
            # Remove non-ASCII characters from list of tokenized words
            word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            # Convert all characters to lowercase from list of tokenized words
            word = word.lower()
            # Remove punctuation from list of tokenized words
            if word == "":
                continue
            # Replace all interger occurrences in list of tokenized words with textual representation
            if word in self.stop_words:
                continue
            #word = self.stemmer.stem(word)
            #word = self.wnl(word, pos='v')
            new_words.append(word)
        return new_words

    def corpusWC(self, Data, wc_res_fil, gram=1):
        self.WC = {}
        for words in Data:
            buff = set([])
            for word in words:
                if word not in self.WC:
                    self.WC[word] = 0
                else:
                    if word not in buff:
                        self.WC[word] += 1
                buff.add(word)
        writer = csv.writer(open(wc_res_fil, 'wt'), delimiter="\t")
        for k,v in sorted(self.WC.items(), key = lambda x:x[1], reverse=True):
            #writer.write("%s\t%d"%(k,v))
            writer.writerow([k,v])

    def BagOfWords(self, Data, vocabulary):
        tmp = []
        for words in Data:
            buff = []
            for word in vocabulary:
                buff.append(words.count(word))
            buff = np.array(buff)
            tmp.append(buff)
        BW = np.array(tmp)
        return BW

    def sliceTopic(self, Y, tSNE1, tSNE2):
        topic = []
        for i, vec in enumerate(Y):
            if vec[0] > tSNE1[0] and vec[0] < tSNE1[1] and vec[1] > tSNE2[0]  and vec[1] < tSNE2[1] :
                topic.append(i)
        return topic

    def showTopic(self, df, topic, Nshow=3):
        count = 0
        keyinfos = []
        for i in topic:
            print(i)
            tw = df.loc[i, "text"]
            time = df.loc[i, "created_at"]
            text = self.Clean(tw)
            print(tw, time)
            print(" ".join(text))
            print()
            if count >= Nshow:
                break
            count += 1
        return df.loc[topic[0]]

    def showTimeline(self, keyinfos):
        keyinfos = sorted(keyinfos, key=lambda x:datetime.strptime(x["created_at"], '%a %b %d %X %z %Y'))
        #print(keyinfos)
        for info in keyinfos:
            print(info["created_at"])
            print(info["text"])
            print()

def AssignCluster(Matrix, Groups):
    res = [[] for i in range(max(Groups)+1)]
    for i,g in enumerate(Groups):
        res[g-1].append(Matrix[i, :])
    return res

def readsencence(df1):
    DC = TwitterCleaner()
    urlsearch = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    tweet_ids = []
    ALL_Sentences = []
    Clean_sentences = []
    for i,row in df1.iterrows():
        sentence = df1.loc[i, "text"].strip("\n")
        sentence = re.sub(r'RT\s@[\w]*', ' ', sentence)
        sentence = re.sub(r'MT\s@[\w]*', ' ', sentence)
        sentence = re.sub(r'@[\w]*', '', sentence)
        #sentence = re.sub(r'#[\w]*', '', sentence)
        sentence = re.sub(r'#', '', sentence)
        sentence = re.sub(r'&amp;', 'and', sentence)
        sentence = re.sub(urlsearch, '', sentence)
        sentence = sentence.strip() 
        sentences = re.split('; |, ',sentence)
        new_sentences = []
        for s in sentences:
            if "..." in s:
                continue
            new_sentences.append(s)
        sentence = "".join(new_sentences)
        if "|" in sentence or "~" in sentence:
            continue
        XX = sentence.split()
        if "I" in XX:
            continue
        if len(XX)<5:
            continue
        if len(XX) > 20:
            continue
        tweet_ids.append(i)
        ALL_Sentences.append(sentence)
        Clean_sentences.append(" ".join(DC.Clean(sentence)))
    return tweet_ids, ALL_Sentences, Clean_sentences

def storyline():
    Keyinfos = []
    for i in range(max(clusters)+1):
        print(i+1)
        topics = np.where(clusters==i)
        #print(topics)
        topics = topics[0]
        info = DC.showTopic(df, topics, Nshow=3)
        Keyinfos.append(info)
        print("-------------")

def LoadWordEmbeddings(fil="../dat/glove/glove.6B.100d.txt"):
    word_embeddings = {}
    f = open(fil, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    return word_embeddings

def Sentence2WE(clean_sentences, word_embeddings):
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    return np.array(sentence_vectors)

def TextRankScoreMat(sentence_vectors):
    sim_mat = np.zeros([len(sentence_vectors), len(sentence_vectors)])
    for i in range(len(sentence_vectors)):
        for j in range(len(sentence_vectors)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
    return sim_mat




######################################################
# Connect to spark ???
######################################################
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


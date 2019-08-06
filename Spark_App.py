#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from collections import namedtuple
from pyspark.sql import SparkSession
from textblob import TextBlob
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import urllib.parse
from lxml import etree
import re


# In[2]:


tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))
def clean_tweet(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    letters_only = re.sub(r'\s+[a-zA-Z]\s+', ' ', letters_only)
    lower_case = letters_only.lower()
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()


# In[3]:


def sentiment_polarity(tweet):
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1


# In[4]:


def sentiment_subjectivity(tweet):
    analysis = TextBlob(clean_tweet(tweet))
    return str(analysis.sentiment.subjectivity)


# In[5]:


sentimentFields = ("text", "cleanText","polarity", "subjectivity")
sentimentObject = namedtuple('sentimentObject', sentimentFields)


# In[6]:


my_spark = SparkSession     .builder     .appName("sparkStreamingApp")     .config("spark.mongodb.input.uri", "mongodb://admin:admin123@127.0.0.1/sparkTwitter.sentiments?authSource=admin")     .config("spark.mongodb.output.uri", "mongodb://admin:admin123@127.0.0.1/sparkTwitter.sentiments?authSource=admin")     .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.11:2.4.0')    .getOrCreate()
sc = my_spark.sparkContext
sc.setLogLevel("ERROR")
ssc = StreamingContext(sc, 10)
ssc.checkpoint("checkpoint_TwitterApp")


# In[7]:


lines = ssc.socketTextStream("127.0.0.1", 5555)


# In[8]:


(lines.reduce(lambda r,d:r+" "+d)
.map(lambda text :(text,clean_tweet(text),sentiment_polarity(text),sentiment_subjectivity(text)))
.map(lambda p: sentimentObject(p[0], p[1], p[2],p[3]))
.foreachRDD(lambda rdd: rdd.toDF().write.format("com.mongodb.spark.sql.DefaultSource").mode("append").save()))


# In[9]:


ssc.start()


# In[11]:


#ssc.stop()


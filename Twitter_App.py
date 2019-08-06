#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import socket
import json


# In[ ]:


class TweetsListener(StreamListener):

  def __init__(self, csocket):
      self.client_socket = csocket

  def on_data(self, data):
    
      try:
            msg = json.loads(data)
            if ('retweeted_status' in msg):
                if ('extended_tweet' in msg['retweeted_status']):
                    #print(msg['retweeted_status']['extended_tweet']['full_text'])
                    self.client_socket.send((str(msg['retweeted_status']['extended_tweet']['full_text']) + "\n").encode('utf-8'))
            elif ('extended_status' in msg):
                #print(msg['extended_status']['full_text'])
                self.client_socket.send((str(msg['extended_status']['full_text']) + "\n").encode('utf-8'))
            else:
                #print(msg['text'])
                self.client_socket.send((str(msg['text']) + "\n").encode('utf-8'))
      except BaseException as e:
          print("Error on_data: %s" % str(e))
      return True

  def on_error(self, status):
      print(status)
      return True


# In[ ]:


ACCESS_TOKEN = 'Enter ACCESS_TOKEN'
ACCESS_SECRET = 'Enter ACCESS_SECRET'
CONSUMER_KEY = 'Enter CONSUMER_KEY'
CONSUMER_SECRET = 'Enter CONSUMER_SECRET'


# In[ ]:


auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)


# In[ ]:


host = "127.0.0.1"
port = 5555
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host,port))
s.listen(2)
c, addr = s.accept()


# In[ ]:


myTweetsListener = TweetsListener(c)
myStream = tweepy.Stream(auth = auth, listener=myTweetsListener)
# Filtring the stream
myStream.filter(languages=["en"],track=['samsung','iphone','huawei','xiaomi','opoo','oneplus'])


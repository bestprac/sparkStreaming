#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession,SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import *
from pyspark.sql.functions import udf,col,when
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np
import pandas as pd


# In[ ]:


my_spark = SparkSession     .builder     .appName("myApp")     .config("spark.mongodb.input.uri", "mongodb://admin:admin123@127.0.0.1/sparkTwitter.sentiments?authSource=admin")     .config("spark.mongodb.output.uri", "mongodb://admin:admin123@127.0.0.1/sparkTwitter.sentiments?authSource=admin")     .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.11:2.4.0')    .getOrCreate()


# In[ ]:


sqlContext = SQLContext(my_spark)
df = my_spark.read.format("com.mongodb.spark.sql.DefaultSource").load()
df.printSchema()


# In[ ]:


sent_df = df.select('cleanText','polarity')


# In[ ]:


# split into training & testing set
splits= sent_df.select(col("cleanText"),col("polarity").alias("label")).randomSplit([0.7,0.3],seed=100)
train= splits[0]
test = splits[1]
train.show()


# In[ ]:


tok = Tokenizer(inputCol="cleanText", outputCol="words")

stopword_rm = StopWordsRemover(inputCol='words', outputCol='words_nsw')

hashingTF = HashingTF(inputCol=stopword_rm.getOutputCol(), outputCol="features")

lr = LogisticRegression(maxIter=10, regParam=0.001)

pipeline = Pipeline(stages=[tok,stopword_rm,hashingTF,lr])


# In[ ]:


model = pipeline.fit(train)


# In[ ]:


prediction = model.transform(test)
selected = prediction.select("cleanText", "probability", "label","prediction")
count=selected.filter('prediction=1 and cleanText like "%huawei%"').count()
print(count)


# In[ ]:


evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(selected)
print("Accuracy = %g " % (accuracy))
print("Test Error = %g " % (1.0 - accuracy))


# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
join_udf=udf(lambda x: " ".join(x))
prediction=prediction.withColumn("worCloud",join_udf(col("words_nsw")))
df=prediction.filter('polarity=1 and (cleanText like "%huawei%" or cleanText  like "%iphone%")').toPandas()


# In[ ]:



word_string=" ".join(df["worCloud"].str.lower())
wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white', 
                      width=1600,
                      height=800
                         ).generate(word_string)


# In[ ]:


plt.clf()
plt.figure(figsize=(20,10),facecolor='k')
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[ ]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# Helper function
def plot_20_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:20]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    plt.figure(figsize=(16, 10))
    plt.bar(x_pos, counts,align='center')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.title('20 most common words')
    plt.show()

count_vectorizer = CountVectorizer(stop_words='english')

count_data = count_vectorizer.fit_transform(df["cleanText"])

plot_20_most_common_words(count_data, count_vectorizer)


# In[ ]:


pos_iphone=prediction.filter('polarity = 1 and cleanText like "%iphone%"').count()
pos_huawei=prediction.filter('polarity = 1 and cleanText like "%huawei%"').count()
pos_xiaomi=prediction.filter('polarity = 1 and cleanText like "%xiaomi%"').count()
pos_samsung=prediction.filter('polarity = 1 and cleanText like "%samsung%"').count()
total=pos_xiaomi+pos_huawei+pos_iphone+pos_samsung
prc_iphone = (pos_iphone/total)*100
prc_huawei = (pos_huawei/total)*100
prc_xiaomi = (pos_xiaomi/total)*100
prc_samsung = (pos_samsung/total)*100


# In[ ]:


import matplotlib.pyplot as plt

labels = 'iphone', 'huawei','xiaomi','samsung'
sizes = [prc_iphone, prc_huawei, prc_xiaomi,prc_samsung]
colors = ['gold', 'yellowgreen', 'lightcoral','lightskyblue']
explode = (0.1, 0, 0,0)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()


# In[ ]:


labels = 'iphone', 'huawei','xiaomi'
sizes = [prcneg_iphone, prcneg_huawei, prcneg_xiaomi]
colors = ['gold', 'yellowgreen', 'lightcoral']
explode = (0.1, 0, 0)  # explode 1st slice

# Plot
plt.clf()
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()


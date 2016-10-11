
## TweetGroups

Twitter is an integral part of marketing and can’t be ignored.  Twitter interactions can not only be a good metric for tracking a marketing campaign’s performance, but it can also be the cause of product or brands success and failure.

In recent years we have all seen examples of bad tweets that have ruined reputations and tarnished brands, so making sure that your company's tweets are throughly planned is essential.  The process of creating and maintaining an effective presence on Twitter is a complex one, but TweetGroups can help you get started:

#### TweetGroups helps to answer two fundamental marketing questions:
1.  What are our market segements (groups of customers)?
2.  How do we engage these customers?

TweetGroups uses the Twitter Search api to return a series of tweets that contain a specific query term (in this example we will use 'Go Pro').  Once the tweets are loaded, TweetGroups clusters all hashtags found in these Tweets via the text of the Tweets that contain them.  In order to do this we will need to follow a few pre-processing steps:

### Text Pre-processing with spaCy

For text processing step we chose to use the Natural Language Processing library spaCy due to a few advantages over other libraries:

1.  Performance: spaCy is written in Cython and contains a wide array of NLP functions that can be parallelized and execute quickly

2.  Flexability: spaCy is an object-oriented approach, so the returned tokens have attributes that are particularly useful for social media data, including the ability to tokenize emojis, parts of speech tagging, and named entity recognition

3.  Usability: spaCy code is clear, concise, well-documented and actively supported

Let's show a quick performance comparison:


```python
import time
import spacy
import timeit
import textacy
%timeit
```


```python
# Loading a pickled list of tweets
import pandas as pd
tweet_list = pd.read_pickle('/Users/nathanbackblaze/DS/Metis/projects/Final Projects/tweet_list_gopro.pkl')
# this list contains about 14000 tweets
```

spaCy's various Parts of Speech attributes can be very helpful when parsing Tweets.  One example of this is that you can use spacy to filter tweets where your company / product name is not the subject of the tweet.  In the example bellow I'll demonstrate how spaCy can help filter out tweets in which 'Uber' is used as a verb


```python
%%timeit
# loading the spacy module
nlp = spacy.en.English()
```

    1 loop, best of 3: 16.2 s per loop


Loading the Spacy English parser does take some time

Now we'll create a fake set of tweets to demonstrate this feature:


```python
demo_tweets = [u'I took an Uber to the other side of town',u'Uber is now launching new features soon',u'Let us uber over to the other side of town']
```


```python
def parse_show_attributes(string):
    parsed = nlp(string)
    for token in parsed:
#   this next line of code will print each word's original form, part of speech tag, and the dependants
        print(token.orth_, token.dep_, token.head.orth_, [t.orth_ for t in token.lefts], [t.orth_ for t in token.rights])
    print
    print "<next tweet>"
    print
```


```python
%%timeit
for tweet in demo_tweets:
    parse_show_attributes(tweet)
```

    (u'I', u'nsubj', u'took', [], [])
    (u'took', u'ROOT', u'took', [u'I'], [u'Uber', u'to'])
    (u'an', u'det', u'Uber', [], [])
    (u'Uber', u'dobj', u'took', [u'an'], [])
    (u'to', u'prep', u'took', [], [u'side'])
    (u'the', u'det', u'side', [], [])
    (u'other', u'amod', u'side', [], [])
    (u'side', u'pobj', u'to', [u'the', u'other'], [u'of'])
    (u'of', u'prep', u'side', [], [u'town'])
    (u'town', u'pobj', u'of', [], [])
    
    <next tweet>
    
    (u'Uber', u'nsubj', u'launching', [], [])
    (u'is', u'aux', u'launching', [], [])
    (u'now', u'advmod', u'launching', [], [])
    (u'launching', u'ROOT', u'launching', [u'Uber', u'is', u'now'], [u'features', u'soon'])
    (u'new', u'amod', u'features', [], [])
    (u'features', u'dobj', u'launching', [u'new'], [])
    (u'soon', u'advmod', u'launching', [], [])
    
    <next tweet>
    
    (u'Let', u'ROOT', u'Let', [], [u'uber'])
    (u'us', u'nsubj', u'uber', [], [])
    (u'uber', u'ccomp', u'Let', [u'us'], [u'over', u'to'])
    (u'over', u'prt', u'uber', [], [])
    (u'to', u'prep', u'uber', [], [u'side'])
    (u'the', u'det', u'side', [], [])
    (u'other', u'amod', u'side', [], [])
    (u'side', u'pobj', u'to', [u'the', u'other'], [u'of'])
    (u'of', u'prep', u'side', [], [u'town'])
    (u'town', u'pobj', u'of', [], [])
    
    <next tweet>
    
    (u'I', u'nsubj', u'took', [], [])
    (u'took', u'ROOT', u'took', [u'I'], [u'Uber', u'to'])
    (u'an', u'det', u'Uber', [], [])
    (u'Uber', u'dobj', u'took', [u'an'], [])
    (u'to', u'prep', u'took', [], [u'side'])
    (u'the', u'det', u'side', [], [])
    (u'other', u'amod', u'side', [], [])
    (u'side', u'pobj', u'to', [u'the', u'other'], [u'of'])
    (u'of', u'prep', u'side', [], [u'town'])
    (u'town', u'pobj', u'of', [], [])
    
    <next tweet>
    
    (u'Uber', u'nsubj', u'launching', [], [])
    (u'is', u'aux', u'launching', [], [])
    (u'now', u'advmod', u'launching', [], [])
    (u'launching', u'ROOT', u'launching', [u'Uber', u'is', u'now'], [u'features', u'soon'])
    (u'new', u'amod', u'features', [], [])
    (u'features', u'dobj', u'launching', [u'new'], [])
    (u'soon', u'advmod', u'launching', [], [])
    
    <next tweet>
    
    (u'Let', u'ROOT', u'Let', [], [u'uber'])
    (u'us', u'nsubj', u'uber', [], [])
    (u'uber', u'ccomp', u'Let', [u'us'], [u'over', u'to'])
    (u'over', u'prt', u'uber', [], [])
    (u'to', u'prep', u'uber', [], [u'side'])
    (u'the', u'det', u'side', [], [])
    (u'other', u'amod', u'side', [], [])
    (u'side', u'pobj', u'to', [u'the', u'other'], [u'of'])
    (u'of', u'prep', u'side', [], [u'town'])
    (u'town', u'pobj', u'of', [], [])
    
    <next tweet>
    
    (u'I', u'nsubj', u'took', [], [])
    (u'took', u'ROOT', u'took', [u'I'], [u'Uber', u'to'])
    (u'an', u'det', u'Uber', [], [])
    (u'Uber', u'dobj', u'took', [u'an'], [])
    (u'to', u'prep', u'took', [], [u'side'])
    (u'the', u'det', u'side', [], [])
    (u'other', u'amod', u'side', [], [])
    (u'side', u'pobj', u'to', [u'the', u'other'], [u'of'])
    (u'of', u'prep', u'side', [], [u'town'])
    (u'town', u'pobj', u'of', [], [])
    
    <next tweet>
    
    (u'Uber', u'nsubj', u'launching', [], [])
    (u'is', u'aux', u'launching', [], [])
    (u'now', u'advmod', u'launching', [], [])
    (u'launching', u'ROOT', u'launching', [u'Uber', u'is', u'now'], [u'features', u'soon'])
    (u'new', u'amod', u'features', [], [])
    (u'features', u'dobj', u'launching', [u'new'], [])
    (u'soon', u'advmod', u'launching', [], [])
    
    <next tweet>
    
    (u'Let', u'ROOT', u'Let', [], [u'uber'])
    (u'us', u'nsubj', u'uber', [], [])
    (u'uber', u'ccomp', u'Let', [u'us'], [u'over', u'to'])
    (u'over', u'prt', u'uber', [], [])
    (u'to', u'prep', u'uber', [], [u'side'])
    (u'the', u'det', u'side', [], [])
    (u'other', u'amod', u'side', [], [])
    (u'side', u'pobj', u'to', [u'the', u'other'], [u'of'])
    (u'of', u'prep', u'side', [], [u'town'])
    (u'town', u'pobj', u'of', [], [])
    
    <next tweet>
    
    (u'I', u'nsubj', u'took', [], [])
    (u'took', u'ROOT', u'took', [u'I'], [u'Uber', u'to'])
    (u'an', u'det', u'Uber', [], [])
    (u'Uber', u'dobj', u'took', [u'an'], [])
    (u'to', u'prep', u'took', [], [u'side'])
    (u'the', u'det', u'side', [], [])
    (u'other', u'amod', u'side', [], [])
    (u'side', u'pobj', u'to', [u'the', u'other'], [u'of'])
    (u'of', u'prep', u'side', [], [u'town'])
    (u'town', u'pobj', u'of', [], [])
    
    <next tweet>
    
    (u'Uber', u'nsubj', u'launching', [], [])
    (u'is', u'aux', u'launching', [], [])
    (u'now', u'advmod', u'launching', [], [])
    (u'launching', u'ROOT', u'launching', [u'Uber', u'is', u'now'], [u'features', u'soon'])
    (u'new', u'amod', u'features', [], [])
    (u'features', u'dobj', u'launching', [u'new'], [])
    (u'soon', u'advmod', u'launching', [], [])
    
    <next tweet>
    
    (u'Let', u'ROOT', u'Let', [], [u'uber'])
    (u'us', u'nsubj', u'uber', [], [])
    (u'uber', u'ccomp', u'Let', [u'us'], [u'over', u'to'])
    (u'over', u'prt', u'uber', [], [])
    (u'to', u'prep', u'uber', [], [u'side'])
    (u'the', u'det', u'side', [], [])
    (u'other', u'amod', u'side', [], [])
    (u'side', u'pobj', u'to', [u'the', u'other'], [u'of'])
    (u'of', u'prep', u'side', [], [u'town'])
    (u'town', u'pobj', u'of', [], [])
    
    <next tweet>
    
    The slowest run took 132.46 times longer than the fastest. This could mean that an intermediate result is being cached.
    1 loop, best of 3: 2.18 ms per loop


With the above example we can see that spaCy was able to identify:
- Uber was the object in the 1st example
- Uber was the subject in the 2nd example
- Uber was a clause in the 3rd example

This might be helpful to filter out noise in an instance where you are trying to determine what users are saying about your firm's actions rather than how they utilize the product

*Note this link helped me translate the part of speech tags http://nlp.stanford.edu/software/dependencies_manual.pdf

We can also detect other named entities in a Tweet, this might be helpful if you wish to know what other products / persons are mentioned in a Tweet (a feature a I would like to add to my project)


```python
demo_tweet = [u'I took an Uber in San Francisco on Monday; it was a Tesla and Elon Musk was there.']
```


```python
entities = list(nlp(unicode(demo_tweet)).ents)
for entity in entities:
    print entity.orth_,entity.label_
```

    Uber GPE
    San Francisco GPE
    Monday DATE
    Tesla GPE
    Elon Musk PERSON


This feature could allow you to map out all the other entities (people / companies) that are mentioned in your tweets.

spaCy also allows you to handle special tokens such as emojis or internet slang, which are extremely relevant for tweets 


```python
slang_tweet = u"lol i luv Uber :)"
for token in nlp(slang_tweet):
    print token.orth_, token.pos_
```

    lol NOUN
    i PRON
    luv VERB
    Uber PROPN
    :) PUNCT


This has a wide array of potential applications, for instance you could use emoji tokens to help with sentiment analysis (since emojis can be clear indicators of sentiment)


```python

```

## NLTK vs. spaCy

Let's show a side-by-side comparison of NLTK vs spaCy , first we will only tokenize the words in the list of tweets:


```python
import nltk
```


```python
%%timeit
for string in tweet_list:
    nltk.tokenize.word_tokenize(string)
```

    1 loop, best of 3: 3.29 s per loop





    <TimeitResult : 1 loop, best of 3: 3.29 s per loop>




```python
%%timeit
for string in tweet_list:
    nlp(string,tag=False,parse=False,entity=False)
```

    1 loop, best of 3: 1.81 s per loop


Let's now compare the much more complex parts of speech tagging


```python
%%timeit
for string in tweet_list:
    nltk.pos_tag(nltk.tokenize.word_tokenize(string))
```

    1 loop, best of 3: 22.3 s per loop



```python
%%timeit
for string in tweet_list:
    nlp(string,tag=True,parse=False,entity=False)
```

    1 loop, best of 3: 2.49 s per loop



```python
import textblob
```


```python
%%timeit
for string in tweet_list:
    textblob.TextBlob(string)
```

    10 loops, best of 3: 85.8 ms per loop


Spacy also allows for multi-threading to help speed up bigger jobs when you have more computational resources available


```python
%%timeit
for doc in nlp.pipe(tweet_list, n_threads = 1, batch_size = 50):
    assert doc.is_parsed
```

    1 loop, best of 3: 12.9 s per loop



```python
%%timeit
for doc in nlp.pipe(tweet_list, n_threads = 8, batch_size = 50):
    assert doc.is_parsed
```

    1 loop, best of 3: 11.8 s per loop


Reference > http://blog.thedataincubator.com/2016/04/nltk-vs-spacy-natural-language-processing-in-python/

## Textacy

Textacy is package built on top of spacy that easily integrates with TFIDF and dimensionality reduction


```python
corpus = textacy.TextCorpus.from_texts('en',tweet_list)
corpus
```




    TextCorpus(2510 docs; 232695 tokens)



The 'Corpus' variable here is a Textacy object not only has the word tokens from our documents, but also has a suite of other attributes, including part of speech tagging.


```python

```

Now that we have our tokenized text matrix, we need to properly weight unimportant words.  Term Frequency Inverse Document Frequency will allow us automatically down-weight words that have less significance in determining the document's topic.

This is especially cruicial in our case.  Because we are taking tweets that match a certain query term ('GoPro'), this means that many of these Tweets will contain the same words such as the query term itself.

In the following step, we are also specifiying that we are tokenizing ngrams, or pairs and triplets of words.  This allows us to tokenize more specific semantic behavior


```python
doc_term_matrix, id2term = corpus.as_doc_term_matrix((doc.as_terms_list(words=True, ngrams=(2,3), named_entities=True)for doc in corpus),weighting='tfidf', normalize=True, smooth_idf=True, min_df=2, max_df=0.95)
```

As a performance demo, here is a quick end to end script that reads the raw text, parses weighted tokens, and outputs the topic models, which might take considerable time with other tools


```python
%%timeit
corpus = textacy.TextCorpus.from_texts('en',tweet_list)
doc_term_matrix, id2term = corpus.as_doc_term_matrix((doc.as_terms_list(words=True, ngrams=(2,3), named_entities=True)for doc in corpus),weighting='tfidf', normalize=True, smooth_idf=True, min_df=2, max_df=0.95)
model = textacy.tm.TopicModel('lsa', n_topics=10)
model.fit(doc_term_matrix)
doc_topic_matrix = model.transform(doc_term_matrix)
```

    1 loop, best of 3: 16.7 s per loop


### Why is Dimensionality Reduction Important?

The text written on social media can be random, arbitrary, and have a wide variety of tokens (including words/phrases/emojis).  Without a way to reduce these high-dimensional token matricies, you can run in to performance issues and clustering may be difficult.  This is particularly import in our case, where we are using clustering algorthims that do not need a n_topics paraments (ie. Affinity Progpogation, DBSCAN, Mean Shift, etc.)




### Latent Semantic Analysis (using Singular Value Decomposition)
LSA is a simple and fast approach to dimensionality reduction.  It works to 'flatten' the dimensionality of the matrix while retaining as much variance as possible, and can work well with unbalanced topics.

LSA does have a few shortcomings, however.  LSA can have trouble in cases where slight variance in words signify a different topic.  For instance, a colleague of mine (Emily) recently built an NLP model using Supreme Court data.  Most of these documents contained a large amount of similar legal jargon, and LSA was unable to effectively parse the latent topics.

Textacy makes switching between dimensionality reduction methods easy, as you are simply passing in the chosen method as a parameter.


```python
model = textacy.tm.TopicModel('lsa', n_topics=10)
model.fit(doc_term_matrix)
doc_topic_matrix = model.transform(doc_term_matrix)
doc_topic_matrix.shape
```




    (2510, 10)



The returned LSA (unlike LDA) values are not necessarily human readable, and don't represent a clear topic distribution.


```python
doc_topic_matrix[0]
```




    array([ 0.05693241, -0.02133441, -0.02307617, -0.02781793,  0.00016982,
           -0.00223301, -0.01919703,  0.02144858,  0.00428661, -0.04453258])




```python
for topic_idx, top_terms in model.top_topic_terms(id2term, top_n=10):
    print('topic', topic_idx, ':', '   '.join(top_terms))
```

    ('topic', 0, ':', u'gopro   $   11.99   camera   video   hero4   rt   gopro $   @gopro   hero')
    ('topic', 1, ':', u'11.99   $   gopro $   $ 11.99   gopro $ 11.99   dog   13.99   harness   fetch   pet')
    ('topic', 2, ':', u'hero4   camera   gopro hero4   silver   hero4 silver   gopro hero4 silver   edition   244.00   hero   gopro hero')
    ('topic', 3, ':', u'scuba   diving   diva   padi   hawaii   sea   mar   underwater   scubadiving   gopro hero4')
    ('topic', 4, ':', u'selfie   @gopro   @xgame @gopro   ion   deftfam   turnup   turnuptuesday   @xgame   texas   austin')
    ('topic', 5, ':', u'drone   quad   fpv   13.99   dji   fly   airquad   atlantic\u2026   naza   turnigy')
    ('topic', 6, ':', u'13.99   $ 13.99   gopro $ 13.99   clamp   flex   adjustable   great   adjustable neck   mount with adjustable   neck')
    ('topic', 7, ':', u'socialmedia   videoshoot   magmabags   denondj   editing   landrover   djlife   route   countryside   dj')
    ('topic', 8, ':', u'gig   download   https://t.co/5zikmckbox   uk   vlog   montage   socialmedia   editing   route   denondj')
    ('topic', 9, ':', u'drone   quad   socialmedia   denondj   editing   magmabags   videoshoot   route   countryside   dj')


### Latent Dirchlet Allocation
Pronounced 'deer_uh_shlay', this method assigns a probability that the document belongs to a given topic making the results more human readable.  LDA is iterative, so it begins by randomly assigning topical distributions and iterates through to optimize the assignments, so it will generally perform slower than LSA.

LDA also has a few limitions.  When using LDA you assume a Dirchlet Prior, which stipulates that the original text documents are written in  the following manner.
- You first decide a set number of words to that the document will have
- You decide on a mixture of topics and draw words from those topics
- The words you use are chosen from each topic 'corpus'

As you can see, this assumption does not necessarily mesh well with how Tweets are actually written.  Tweets are approximately all of the same length (~ 140 characters), and you may only really be Tweeting about one topic, not choosing from a distribution.

Furthermore, LDA is designed to find topics that are 'furthest' from each other, so LDA can have trouble identifying uneven topic distributions (which is often the case on Twitter).

As a result, LDA is often better suited for understanding the distribution of a topics in the given corpus, rather than specifically classifying each document

LDA requires and interger matrix, so TFIDF cannot be used:


```python
doc_term_matrix_tf, id2term = corpus.as_doc_term_matrix((doc.as_terms_list(words=True, ngrams=(2,3), named_entities=True)for doc in corpus),weighting='tf')
```

    /Users/nathanbackblaze/anaconda/lib/python2.7/site-packages/sklearn/utils/validation.py:420: DataConversionWarning: Data with input dtype int64 was converted to float64 by the normalize function.
      warnings.warn(msg, DataConversionWarning)



```python
doc_term_matrix_tf
```




    <2510x37158 sparse matrix of type '<type 'numpy.float64'>'
    	with 102213 stored elements in Compressed Sparse Row format>




```python
%%timeit
model = textacy.tm.TopicModel('lda', n_topics=10)
model.fit(doc_term_matrix_tf)
doc_topic_matrix_tf = model.transform(doc_term_matrix_tf)
doc_topic_matrix_tf.shape
```




    (2510, 10)




```python
doc_topic_matrix_tf[0]
```




    array([ 0.10000046,  0.1000001 ,  0.10000051,  0.10000011,  0.10000051,
            0.10000021,  0.10000032,  0.10000075,  3.97297869,  0.10000169])




```python
for topic_idx, top_terms in model.top_topic_terms(id2term, top_n=10):
    print('topic', topic_idx, ':', '   '.join(top_terms))
```

    ('topic', 0, ':', u'gopro   13.99   morning   $ 13.99   $   gopro $ 13.99   gopro $   happy   travel   rt')
    ('topic', 1, ':', u'gopro   3dprint   give   \U0001f440   blackdress   link   cape   littleblackridinghood   pullup   vintage')
    ('topic', 2, ':', u'gopro   2016   vscocam   vsco   brasil   camera   care   panda   studio   landrover')
    ('topic', 3, ':', u'gopro   europe   fly   agario   turkey   fake   slitherio   turkiye   afk   oyun')
    ('topic', 4, ':', u'gopro   @gopro   drone   fun   rt   quad   video   parkour   freerunning   @goprouk')
    ('topic', 5, ':', u'gopro   summer   travel   thailand   incredible   video   photography   blogger   leh   ladakh')
    ('topic', 6, ':', u'gopro   travel   gopro\u306e\u3042\u308b\u751f\u6d3b   @gopro   oahu   nassau   dallas   goprohero4   hawaii   calpe')
    ('topic', 7, ':', u'gopro   venice   munich   snorkel   best   today   \u6c96\u7e04   edit   dji   best sale')
    ('topic', 8, ':', u'gopro   rt   video   goprohero4   hero4   @gopro   camera   gopro\u2026   hero   summer')
    ('topic', 9, ':', u'gopro   selfie   \u5bcc\u58eb\u5c71   @gopro   download   2016   music   gig   vlog   lens')


### Non-Negative Matrix Factorization (NMF)
True to it's name, this newer method requires that the given matrix be non-negative.  While similar to LDA, NMF has more restrictive paramenters.  This leads to less flexibility, but also allows for an improvement of performance over LDA and works well out of the box on short texts (like tweets).

One noted downside of NMF is that results can vary wildly with a slight change of n_topics.  While the topic distribution may not change significantly with LDA when the n_topics are increased, the NMF topic distributio may go from focused to incoherent.  This is particularily an issue with the clustering methods we will be using as they may have a large number of topics.




```python
doc_term_matrix, id2term = corpus.as_doc_term_matrix((doc.as_terms_list(words=True, ngrams=(2,3), named_entities=True)for doc in corpus),weighting='tfidf', normalize=True, smooth_idf=True, min_df=2, max_df=0.95)
```


```python
%%timeit
model = textacy.tm.TopicModel('nmf', n_topics=10)
model.fit(doc_term_matrix)
doc_topic_matrix = model.transform(doc_term_matrix)
doc_topic_matrix.shape
```

    1 loop, best of 3: 264 ms per loop
    100 loops, best of 3: 7.95 ms per loop





    (2510, 10)




```python
doc_topic_matrix[0]
```




    array([ 0.05693241, -0.02133441, -0.02307617, -0.02781793,  0.00016982,
           -0.00223301, -0.01919703,  0.02144858,  0.00428661, -0.04453258])




```python
for topic_idx, top_terms in model.top_topic_terms(id2term, top_n=10):
    print('topic', topic_idx, ':', '   '.join(top_terms))
```

    ('topic', 0, ':', u'gopro   rt   @gopro   video   hero   camera   win   gopro hero   win a gopro   enter')
    ('topic', 1, ':', u'hero4   gopro hero4   silver   $   hero4 silver   gopro hero4 silver   244.00   244   edition   camera')
    ('topic', 2, ':', u'11.99   gopro $ 11.99   $ 11.99   gopro $   dog   $   harness   fetch   pet   pet chest')
    ('topic', 3, ':', u'scuba   diving   diva   padi   hawaii   sea   mar   underwater   scubadiving   buceo')
    ('topic', 4, ':', u'goprooftheday\u2026   follzqome   likesforlikes   fothograpy   goprohero3   follow   like   gopro   share   climbing')
    ('topic', 5, ':', u'ladakh #   https://t.co/kdtqqyx2cx   leh   blogger   india   ladakh   people   incredible   blog   place')
    ('topic', 6, ':', u'gig   download   https://t.co/5zikmckbox   uk   vlog   montage   vlogs   heavymetal   download gopro montage   @gopro @goprouk https://t.co/5zikmckbox')
    ('topic', 7, ':', u'artfood\U0001f3a8   \u6df1\u5733   shenzhen   luohu   \u5e7f\u4e1c\u2026   \u7f57\u6e56   guangdong   party   https://t.co/idcpnh9kec   \u5e7f\u4e1c\u2026 https://t.co/idcpnh9kec')
    ('topic', 8, ':', u'13.99   $ 13.99   gopro $ 13.99   gopro $   $   clamp   flex   adjustable   mount   great')
    ('topic', 9, ':', u'\u5bcc\u58eb\u5c71   \u7a7a\u64ae   \u6cb3\u53e3\u6e56   https://t.co/gxgfe547yv   gopro https://t.co/gxgfe547yv   \u672c\u6816\u6e56   \u5bcc\u58eb\u5c71 https://t.co/dyase8h0me   interpose+   \u4e16\u754c\u907a\u7523 \u5bcc\u58eb\u5c71   https://t.co/dyase8h0me')




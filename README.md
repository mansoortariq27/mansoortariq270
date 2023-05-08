
# Project Title

Quora Question Similarity Detections
![WhatsApp Image 2023-05-03 at 11 08 43 PM (1)](https://user-images.githubusercontent.com/98485769/236951122-d4fc64ae-cc25-4295-a4d4-4effe37b7a79.jpeg)


## Introduction

The Document Similarity project attempts to create an algorithm that can precisely assess how similar two Quora questions are to one another. This algorithm will assist users in locating queries that are redundant or similar, enhancing the site's quality and relevance of its information. The fact that Quora questions might be worded differently even when they are requesting the same information presents one of the biggest hurdles for this project.
In order to effectively detect comparable queries, the algorithm must take syntax, semantics, and context into account. By making the process of asking and answering questions more efficient, this project will improve the Quora user experience.
## DataSet

A collection of question pairs with a binary label indicating whether or not they have the same meaning can be found in the Quora Question Pairs dataset on Kaggle. The question text and individual question identifiers are included in the dataset. With the aid of this dataset, a machine learning model that can precisely detect duplicate questions on the Quora platform will be created. This model will assist to enhance user experience and avoid redundancy. The construction of deep learning models for semantic similarity and challenges involving natural language processing have both made use of this dataset (Kaggle, 2017).
## Preparation of DataSet

First, a collection of stopwords is generated using the NLTK package in order to prepare the dataset. Then, "preprocess()" is defined as a special function to preprocess the text in each row of the DataFrame. In addition to removing stopwords and punctuation, this function tokenizes the text and merges the tokens back into a string. The "apply_parallel()" method applies the "preprocess()" function to each row of the DataFrame utilising parallel processing with 4 processes, accelerating the text preparation process. This makes it possible to process time for handling massive datasets much faster. The dataset is prepared for additional analysis and modelling when text preparation is complete.
# Implementation of Models

##  Deliverable 1

### K-Means:

The dataset's text data is first consolidated into a single column from two other columns. The text data is then subjected to a TF-IDF vectorizer to convert it to a numerical representation. The MiniBatchKMeans method is then repeated with various values of k using the altered data. A list contains the algorithm's inertia for each value of k.

How far out from a cluster's centre its individual points are is determined by the cluster's inertia. The ideal number of clusters is determined using the elbow approach. Plotting the elbow curve places the inertia on the y-axis and the number of clusters on the x-axis. The elbow curve's point where the rate of inertia loss slows down to create an elbow shape is where the ideal number of clusters should be located.

A red dashed line on the elbow curve designates the number 8 as the ideal number of clusters. The clustering performance reaches this value of k when adding more clusters no longer significantly improves it.

The TF-IDF matrix of the text data is subjected to MiniBatchKMeans clustering with various numbers of clusters using this code. The clusters are then shown in two dimensions after PCA is used to decrease the matrix's dimensionality. This makes it possible to see how the text data is grouped visually. From 1 to 9, there are different numbers of clusters, and for each number, the clusters are presented in a scatter plot. The scatter plots display the clusters in two dimensions, with a unique colour for each cluster. The plots may be used to assess the effectiveness of the clustering and identify the number of clusters that best groups the text data.

### Cosine Similarity:

We then combined the question1 and question2 columns into a single corpus before computing the cosine similarity between pairs of questions in the Quora Question Pairs dataset. The corpus was then converted into a bag-of-words (BOW) format using the CountVectorizer class from the scikit-learn package. Because of this, I was able to see each question as a vector of word counts. The cosine_similarity function from the metrics package of scikit-learn was then used to determine the cosine similarity of the BOW vectors for each pair of questions.

We employed multithreading to parallelize the procedure since calculating the cosine similarity for all question pairings is a computationally demanding effort. We created a method called calculate_similarity that receives an index i as input, extracts the BOW vectors for the associated question pair, and determines how similar they are to one another in terms of cosine. Then, we executed several versions of this function concurrently on various data subsets using the ThreadPoolExecutor class from the concurrent.futures package. Based on the capabilities of my system, we decided on the number of worker threads.

The results of the cosine similarity computations were kept in a list, which was later changed into a NumPy array and appended to the original dataframe as a new column with the name "cosine_similarity." This column was used as the target variable for machine learning models that aimed to predict whether a pair of questions were duplicates.


## Deliverable 2

### Bert Model

The development of Natural Language Processing (NLP) in recent years has made it possible for robots to comprehend human language and provide insightful data from it. Representing text data in a machine-readable manner is a major obstacle in NLP. The answer to this issue is to transform the text input into a numerical vector, which computer learning models can quickly process. Word embedding is the process of transforming text data into numerical vectors.

Many NLP applications, such as text categorization, sentiment analysis, and question-answering systems, make extensive use of word embeddings. Word embeddings do not fully capture the context and meaning of the complete phrase, hence they have limits. Sentence embeddings were developed to get around this restriction.

The content and context of the full phrase are captured by sentence embeddings, a numerical representation of a sentence. For the purpose of creating sentence embeddings, the BERT (Bidirectional Encoder Representations from Transformers) model has been extensively employed. We will go through how to create sentence embeddings using the BERT model in this report.


### Data Preprocessing

Preprocessing the data is the initial stage in creating sentence embeddings. Tokenization, which entails dividing the text into separate tokens or words, is a step in the data preparation process. WordPiece tokenization, used by the BERT model, divides the words into subwords.

The BERT tokenizer may be used to tokenize data and turn it into numerical vectors after that. Each token is associated with an index in the BERT vocabulary, which has over 30,000 words, using the BERT tokenizer. The BERT tokenizer additionally inserts unique tokens, [CLS] and [SEP], to the start and end of the sentence, respectively. The [CLS] token denotes the beginning of the sentence, and the [SEP] token denotes the break between two consecutive sentences.

### Batching and Dataloader

The next stage is to organise data into batches that can be processed quickly. We can handle numerous inputs at once thanks to batching, which accelerates model training. The data is loaded into the model in batches using the dataloader. After receiving the input data, the dataloader outputs batches of data with a predetermined batch size.

The method make_batches in the provided code produces a dataloader with a batch size of 32. The function returns a dataloader and accepts input_ids and attention_masks as input. The tokens are represented numerically by input_ids, and attention_masks are used to instruct the model which tokens are significant and which are not.

### Generating Sentence Embeddings

The BERT model is used to create sentence embeddings as the last stage. Sentence embeddings are produced using the output from the last layer of the BERT model, which has several levels. A list of concealed states for each token in the input phrase is the result of the last layer. We must extract the hidden state of the [CLS] token, which symbolises the meaning of the entire sentence, in order to produce sentence embeddings.

The calculate_embeddings function is utilised in the provided code to produce sentence embeddings. The method returns a list of sentence embeddings after receiving a dataloader as input. Each batch of data is iterated over by the function before being sent to the BERT model. The [CLS] token is extracted from the output of the final layer, which represents the sentence's meaning. The function then returns a list of sentence embeddings.

### Calculating cosine similarity

In order to compare the BERT embeddings of the input text, the cosine similarity is determined. The cosine similarity, which is determined as the cosine of the angle between two non-zero vectors, is a metric for how similar they are. The vectors are said to be identical if their cosine similarity is 1, and they are said to be entirely distinct if it is 0.

### Accuracy computation

The cosine similarity scores and the ground truth labels are used to compare the accuracy of the text similarity calculation. The 'is_duplicate' labels from the example dataset, which determine if two bits of text are identical, serve as the ground truth labels in this code implementation.

## Conclusion

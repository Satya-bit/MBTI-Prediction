# MBTI-Prediction
This paper introduces a machine learning approach to predict Myers-Briggs Type Indicator (MBTI) personality types based on survey responses. By leveraging advanced algorithms, our model accurately classifies individuals into their respective personality types, contributing to the intersection of psychology and artificial intelligence.

# INTRODUCTION
The MBTI personality prediction project utilizes machine learning methodologies to predict personality types based on individuals' behavioral patterns and preferences. The MBTI divides individuals into 16 personality types using four pairs of characteristics. By analyzing a dataset of MBTI survey responses, the machine learning model is trained to accurately classify individuals into one of the 16 personality types. This project aims to deepen our understanding of human behavior and has potential applications in psychology, team dynamics, and personalization. It showcases the intersection of psychology and artificial intelligence, highlighting the predictive capabilities of machine learning in categorizing human personality traits.
The Myers-Briggs Type Indicator (MBTI) sorts people into one of 16 personality types, determined by four dimensions:
•INTROVERSION (I) – EXTROVERSION (E)
• INTUITION (N) – SENSING (S)
• THINKING (T) – FEELING (F)
• JUDGING (J) – PERCEIVING (P)

![image](https://github.com/Satya-bit/MBTI-Prediction/assets/70309925/5a681e1d-a357-4ef2-ab06-22d249760ec1)

# DATASET DESCRIPTION
The dataset focuses on exploring the Myers Briggs Type Indicator (MBTI), a widely used personality type system categorizing individuals into one of 16 distinct types based on preferences in four axes: Introversion/Extroversion, Intuition/Sensing, Thinking/Feeling, and Judging/Perceiving. This system, rooted in Carl Jung's cognitive functions theory, has enjoyed extensive usage across various domains including businesses, online platforms, research, and personal interest. Despite its popularity, the MBTI's validity has faced scrutiny due to inconsistencies in experimental results. The dataset aims to investigate if any discernible patterns exist between specific MBTI types and writing styles, thereby contributing to the ongoing discourse surrounding the test's efficacy in predicting and categorizing behavior.
The dataset consists of over 8600 rows, with each row representing a person's MBTI type (a 4-letter code denoting their personality type) along with a section containing the last 50 posts made by that person. The posts are separated by "|||" (3 pipe characters).

![image](https://github.com/Satya-bit/MBTI-Prediction/assets/70309925/3cb0255d-2883-4ee0-ac0c-a97931aef619)


# EXPLORATORY DATA ANALYSIS
We meticulously examined the dataset's composition and delved into the distribution of personality types. Through correlation and comparative analyses, we unveiled intriguing patterns and insights into behavioral trends across different MBTI types. While mindful of limitations, our findings fuel our quest for deeper understanding, guiding future investigations into the rich tapestry of human personality dynamics.

![image](https://github.com/Satya-bit/MBTI-Prediction/assets/70309925/173572d2-6b87-42e9-8ff3-7c8e64b78d78)

![image](https://github.com/Satya-bit/MBTI-Prediction/assets/70309925/03b1836f-4692-41c8-9a37-97e97cb0af1f)

![image](https://github.com/Satya-bit/MBTI-Prediction/assets/70309925/97b48d8e-fda9-48b6-adce-41606166d637)

Majority of the posts have positive sentiment.


![image](https://github.com/Satya-bit/MBTI-Prediction/assets/70309925/546c4fa8-8fac-429b-95df-9a99fefb6be8)

# DATA CLEANING AND PREPROCESSING
We clean text like removing hyperlinks, spaces using regex also removed stop words using NLP library. We also performed tokenization and padding.

=>Initial Tweet
![image](https://github.com/Satya-bit/MBTI-Prediction/assets/70309925/d01ec40b-a95e-4620-960e-3b6f3e69a37b)

=> After cleaning
![image](https://github.com/Satya-bit/MBTI-Prediction/assets/70309925/9b1c70ef-335f-42ec-acce-7a2c42208cae)

#HANDLING IMBALANCE DATASET

=>RANDOM OVERSAMPLING
To address class imbalance in our project on MBTI personality prediction, we've implemented machine learning strategies such as random oversampling. Random oversampling involves augmenting instances in the minority class until it matches the count of the majority class. Unlike SMOTE, which generates new instances based on nearest neighbours, we opt to duplicate instances from the minority class randomly. However, we're cautious about potential pitfalls; careless application may lead to overfitting as it introduces redundancy and may accentuate specific patterns unique to the minority class.

![image](https://github.com/Satya-bit/MBTI-Prediction/assets/70309925/b2f1419b-a9fb-45d4-9743-55b1036c045f)

![image](https://github.com/Satya-bit/MBTI-Prediction/assets/70309925/f62991f3-7d02-468d-838d-eef4559b9266)



# WORD EMBEDDINGS

=>FAST TEXT
Facebook's AI Research group developed the freely available library FastText, which is a potent tool for effective text classification and learning representations. It works with the idea of word embeddings, where words are represented as dense vectors in a continuous space. FastText stands out due to its capacity to create embeddings for character n-grams as well as individual words. This allows it to process new words more effectively and understand morphological subtleties than traditional word-based embeddings. FastText is widely used for applications like as sentiment analysis, language modeling, and text classification; it is especially well-suited for contexts with constrained resources and big datasets "Crawl-300d-2M" is a pre-trained word embedding model that was developed using a large text corpus that was crawled across the internet. Compact vector representations of words used in machine learning that capture semantic relationships based on contextual usage are called word embeddings. A 300-dimensional vector is used to represent each word, as indicated by the "300d" symbol. The "2M" stands for around two million unique words or tokens, which is the dataset that the model was trained on. In many natural language processing applications, such as text classification, sentiment analysis, and machine translation, these word embedding models are used as feature extractors.

![image](https://github.com/Satya-bit/MBTI-Prediction/assets/70309925/2537d5ee-f414-4b1f-8935-5007599c2988)

# COVOLUTIONAL NEURAL NETWROK
For text classification, a convolutioal neural network (CNN) architecture is used in the sequential model. It starts with an Embedding layer that uses non-trainable 300-dimensional pre-trained word embeddings. ReLU activates two Conv1D layers, each of which has 256 filters with a kernel size of 5, for the purpose of extracting features. Layers of MaxPooling1D with a pool size of 5 are then applied. GlobalMaxPooling1D further reduces the dimensionality of the feature map. Len(label_encoder.classes_) determines the classification probabilities for each class in the final Dense layer, which is activated by softmax. For added abstraction, two more Dense layers are added, each having 256 neurons and triggered by ReLU. With sparse categorical cross-entropy loss, the model is put together.

![image](https://github.com/Satya-bit/MBTI-Prediction/assets/70309925/25d1f1f1-b320-4a19-bf6a-76ab915520c6)

# BIDIRECTIONAL ENCODER REPRESENTATIONS FROM TRANSFORMERS(BERT)
![image](https://github.com/Satya-bit/MBTI-Prediction/assets/70309925/3d1c739c-59e2-4deb-8e7c-c06414355c67)

•A TensorFlow model is configured with the inputs and outputs. The input_word_ids layer is the value of the inputs parameter, while the output of the Dense layer (pred) is the value of the outputs parameter.
•Loss Function: The categorical_crossentropy loss function is used when compiling the model, and it is suitable for multi-class classification problems in which the target labels are given in a format that is singularly encoded.
•Optimizer: A learning rate of 0.00001 is applied while using the Adam optimizer. Because of its capacity for flexible learning rate, Adam is a well-liked option for deep learning model training.
•Metrics: A statistic called accuracy is used to assess how well the model performs both during training and validation.

For a multi-class classification task, this function builds a BERT-based model that is specially customized. With the help of a basic neural network classifier and the potent feature extraction capabilities of BERT, it can forecast class probabilities. We apply the normal procedure in BERT-based models for sentence-level tasks to the categorization of the [CLS] token using its embedding. Various BERT models were tested, including tiny, medium, tiny, mini, small, and BERT-based. There is less space needed for this compact BERT variant. Bert-base-uncased was our last attempt, and it produced an accuracy of 69%.

# ACCURACY

![image](https://github.com/Satya-bit/MBTI-Prediction/assets/70309925/9f88c9f1-79f3-4178-8efa-3b9dff25df89)

 We got high testing accuracy because we were having highly imbalanced dataset. Though we tried to handle imbalance datset but it was difficult and didn't improve much of the accuracy


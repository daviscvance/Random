# Newsela MLE

Prompt: What words or phrases appear more frequently in questions that students tend to do poorly on, and what appear more frequently in questions that students do well on?

The data comes in as a text question with a score of the percentage of correct responses (multiple choice). It was appropriate to say scores >50% were good, and the rest performed less than well, but I put the threshold at 58% to have more evenly distributed classes.

I approached the problem with Tf-Idf (Term frequency - Inverse document frequency) and CountVectorizer to grab uni, bi, and trigrams and to remove stop_words of common phrases e.g. "Read the sentence from the passage".

Using a Naive Bayes classifier, I wanted to pull the most and least salient terms associated with each of the classes. The Tf-Idf model performed the same as the count vec model (both were 58% or slightly better than a coin flip due to the even classes) so I finished analysis with the CV data. The important features contributing to the model most meaningfully should also be the most frequent terms and we can reasonable define which words best match the strong/poor classes with this model. Later I found there was nothing helpful about using a naive bayes algorithm and switched to pandas analysis to collect frequent terms.

The most frequently used words and phrases in performant questions, outside of some common phrasing, are:
word, except, used, meaning, contains, important, people, sentences, reason, include

The most frequently used words and phrases in poorly performing questions, outside of some common phrasing, are: 
except, word, author, sentences, contains, important, people, include, reason, evidence

The difference being some ordering and the terms: 

-   used and meaning for high ranking (where these were not in the low ranking top 10)
-   author and evidence for low ranking (where these were not in high ranking top 10)

Running the `quiz_questions.py` scripts will yield the following output:

========== Count Vectorization ======================================================================
Accuracy score for Naive Bayes: 0.5811672388425699

 Low score Top10 phrases: ['word' 'except' 'used' 'meaning' 'contains' 'important' 'people'
 'sentences' 'reason' 'include']

 Low score Bot10 phrases: ['use point' 'present' 'states primary' 'contains idiomatic phrase'
 'sufficient' 'add' 'kind evidence' 'claim made' 'stocker' 'pro least']

 High score Top10 phrases: ['except' 'word' 'author' 'sentences' 'important' 'contains' 'people'
 'include' 'evidence' 'used']

 High score Bot10 phrases: ['last year' 'california kingsnakes' 'friends' 'percentage'
 'whooping cranes' 'whooping' 'mcfadden' 'damage' 'guns' 'cranes']

 ========== Count Vectorization ======================================================================
success_lvl   high              low           high_rank low_rank
               sum      mean    sum      mean                   
word         651.0  0.126580  439.0  0.086965       1.0      2.0
except       324.0  0.062998  541.0  0.107171       2.0      1.0
used         289.0  0.056193  201.0  0.039818       3.0     11.0
meaning      266.0  0.051721  136.0  0.026941       4.0     25.0
contains     254.0  0.049388  286.0  0.056656       5.0      5.0
important    249.0  0.048415  271.0  0.053685       6.0      6.0
people       239.0  0.046471  250.0  0.049525       7.0      7.0
sentences    236.0  0.045888  329.0  0.065174       8.0      4.0
reason       227.0  0.044138  212.0  0.041997       9.0      9.0
include      219.0  0.042582  236.0  0.046751      10.0      8.0
success_lvl   high              low           high_rank low_rank
               sum      mean    sum      mean                   
except       324.0  0.062998  541.0  0.107171       2.0      1.0
word         651.0  0.126580  439.0  0.086965       1.0      2.0
author       173.0  0.033638  351.0  0.069532      14.0      3.0
sentences    236.0  0.045888  329.0  0.065174       8.0      4.0
contains     254.0  0.049388  286.0  0.056656       5.0      5.0
important    249.0  0.048415  271.0  0.053685       6.0      6.0
people       239.0  0.046471  250.0  0.049525       7.0      7.0
include      219.0  0.042582  236.0  0.046751      10.0      8.0
reason       227.0  0.044138  212.0  0.041997       9.0      9.0
evidence      84.0  0.016333  206.0  0.040808      38.0     10.0

Run these commands in order to reproduce with the newsela.zip contents:
python3 -m pip install --user virtualenv
python3 -m venv newsela_env
source newsela_env/bin/activate
pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 quiz_questions.py

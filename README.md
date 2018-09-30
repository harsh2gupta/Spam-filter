# Spam-filter

Task: Implement the Naive Bayes algorithm to classify spam.

Details (from the project page http://www3.cs.stonybrook.edu/~cse537/project05.html):
The dataset we will be using is a subset of 2005 TREC Public Spam Corpus. 
It contains a training set and a test set. Both files use the same format: 
each line represents the space-delimited properties of an email, with the first 
one being the email ID, the second one being whether it is a spam or ham (non-spam), 
and the rest are words and their occurrence numbers in this email. 
In preprocessing, non-word characters have been removed, and features selected similar 
to what Mehran Sahami did in his original paper using Naive Bayes to classify spams.

<h1>CS F469 - Information Retrieval</h1>
<h2>Assignment – 2: Recommender Systems</h2>
<b>Language:	Python 3.5</b>

<h2>Contributors:</h2>
<li>Mahicharan Galla		2016A7PS0054H</li>
<li>Dhulipudi Avinash		2016A7PS0113H</li>
<li>Bharath KNS			2016A7PS0103H</li>
<li>M Tejo Vardhan			2016AAPS0150H</li>

<h2>Aim:</h2>
To implement different recommender system models using the collaborative filtering method (with and without baseline approach), singular value decomposition (SVD) and CUR algorithms.

<h2>Working:</h2>

1.	The data which was extracted from https://grouplens.org/datasets/movielens/ is pre-processed by the preprocess.py file which creates and saves the sparse matrix A and test data in the same directory.
2.	Collaborative Filtering on the sparse matrix A and predicting the test data is done by running collaborative_filtering.py.
3.	Collaborative Filtering with Baseline approach on the sparse matrix A and predicting the test data is done by running collaborative_filtering_baseline.py.
4.	Singular Value Decomposition (SVD) approach to predicting matrix A is done by running svd.py. This includes both 100% energy retention model and the 90% energy retention model.
5.	CUR decomposition approach with 100% energy retention is done on sparse matrix A by running cur.py.
6.	CUR decomposition approach with 90% energy retention is done on sparse matrix A by running cur_90.py.

<h2>Installation:</h2>

To run the following code, Anaconda needs to be readily installed.
<li>Anaconda can be installed by following the following link: https://docs.anaconda.com/anaconda/install/</li>

Twitter provides a service for posting short messages. In practice, many of the
tweets are very similar to each other and can be clustered together. By
clustering similar tweets together, we can generate a more concise and organized
representation of the raw tweets, which will be very useful for many Twitter-
based applications (e.g., truth discovery, trend analysis, search ranking, etc.)

In this assignment, you will learn how to cluster tweets by utilizing Jaccard
Distance metric and K-means clustering algorithm. Objectives: Compute the
similarity between tweets using the Jaccard Distance metric. Cluster tweets
using the K-means clustering algorithm.

Implement the tweet clustering function using the Jaccard Distance metric and
K-means clustering algorithm to cluster redundant/repeated tweets into the same
cluster. You are free to use any language or package, but it should be clearly
mentioned in your report. Note that while the K-means algorithm is proved to
converge, the algorithm is sensitive to the k initial selected cluster centroids
(i.e., seeds) and the clustering result is not necessarily optimal on a random
selection of seeds. In this assignment, we provide you with a list of K initial
centroids that have been tested to generate good results.

How to run?
Language : Python3 and above
Libary: urllib Ex: pip install urllib

python kmeans_tweets.py <numberOfClusters> <initialSeedsFile> <input-file-name> <output-file-name>
       
python kmeans_tweets.py 25 InitialSeeds.txt Tweets.json tweets-k-means-output.txt

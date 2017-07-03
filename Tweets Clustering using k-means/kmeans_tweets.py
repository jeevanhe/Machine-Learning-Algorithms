import sys
import json
from urllib.request import urlopen, Request

tweets_list = []
initTweets = []

class Tweet:
    text = ""
    id_str = ""

def tweet_define(ind_line):
    obj = Tweet()
    obj.__dict__.update(ind_line)
    return obj

class Cluster:
    def __init__(self, points):
        # Points belonging to the cluster
        self.points = points
        # Set up the initial centroid (this is usually based off one point)
        self.centroid = self.calCentroid()

    def __repr__(self):
        return str(self.points)

    def update(self, points):
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calCentroid()
        shift = getJaccardDistance(old_centroid, self.centroid)
        return shift

    def calCentroid(self):
        least_value = 0.0
        # Get a list of all coordinates in this cluster
        for p in self.points:
            new_centroid = p
            sum_values = 0.0
            for tp in self.points:
                sum_values += getJaccardDistance(p, tp)
            if sum_values <= least_value:
                least_value = sum_values
                new_centroid = p
            return  new_centroid


def get_sse(clustersList):
    list_sse = []
    itr = 0
    total_sse = 0.0
    for i in clustersList:
        sum1 = 0.0
        for j in i:
            term1 = (getJaccardDistance(list_of_centroids[itr].centroid, j) ** 2)
            sum1 += term1
        itr += 1
        list_sse.append(sum1)
    total_sse = sum(list_sse)
    print("Number of clusters - ", int(sys.argv[1]))
    print("Validation SSE - ", total_sse)


def kmeans_cluster_algo(tweets, default_centroids, cutoff):
    iteration = 1
    initial = []
    for id in default_centroids:
        for t in tweets:
            if t.id_str == id:
                initial.append(t)

    # Create k clusters using those centroids
    clusters = [Cluster([p]) for p in initial]
    loopCounter = 0
    while (iteration <= 25):
        # Create a list of lists to hold the points in each cluster
        lists = [[] for c in clusters]
        clusterCount = len(clusters)
        loopCounter += 1
        # For every point in the dataset
        for p in tweets:
            smallest_distance = getJaccardDistance(p, clusters[0].centroid)
            clusterIndex = 0
            for i in range(clusterCount - 1):
                distance = getJaccardDistance(p, clusters[i + 1].centroid)
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i + 1
            lists[clusterIndex].append(p)
        maxShift = 0.0

        for i in range(clusterCount):
            shift = clusters[i].update(lists[i])
            maxShift = max(maxShift, shift)
        if maxShift < cutoff:
            print("Converged after %s iterations" % loopCounter)
            break
        iteration += 1
    global list_of_centroids
    list_of_centroids = clusters
    return lists


def getJaccardDistance(str1, str2):

    str1 = str1.text
    str2 = str2.text
    str1 = str1.split()
    str2 = str2.split()
    union = list(set(str1 + str2))
    intersection = list(set(str1) - (set(str1) - set(str2)))
    jaccard_dist = float(len(intersection)) / len(union)
    return 1 - jaccard_dist

def save_result(out_file, num_clusters, list_of_clusters):

    file = open(out_file, 'w+')
    file.write("Cluster-ID    Tweets Cluster List\n")
    for i in range(num_clusters):
        file.seek(0, 2)
        str2 = str(i + 1) + "      " + str([cluster.id_str for cluster in list_of_clusters[i]])
        file.write(str2)
        file.write("\n")
    file.close()

if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Invalid input!")
        print("Correct input :<num_clusters> <initialSeedsFile> <input-file-name> <output-file-name>")
        exit(0)
    # get clustering value k
    num_clusters = int(sys.argv[1])
    # get initial seed file
    initial_seed_file = sys.argv[2]
    # test file URL
    input_file_url = sys.argv[3]
    # output file
    output_file = sys.argv[4]

    # reading list of all tweets and mapping them into Tweet objects
    req = Request(input_file_url, headers={'User-Agent': 'Mozilla/5.0'})
    url_data = urlopen(req)
    url_data = url_data.readlines()
    for tweet in url_data:
        tweetobj = json.loads(tweet.decode('utf-8'), object_hook=tweet_define)
        tweets_list.append(tweetobj)

    # reading list of initial tweets
    req2 = Request(initial_seed_file, headers={'User-Agent': 'Mozilla/5.0'})
    file = urlopen(req2)
    for line in file:
        line = line.decode('utf-8')
        line = line.replace(",\n", "")
        initTweets.append(line)
    opt_cutoff = 0.01
    # exit if given k value is greater than number of coordinate points
    if num_clusters > len(tweets_list):
        print ("The number of Tweets  is ", len(tweets_list))
        print ("Make sure num_clusters <= ", len(tweets_list))
        exit(0)

    if num_clusters > 25:
        print ("Enter k= 25 by default")
        exit(0)

    clustersList = kmeans_cluster_algo(tweets_list, initTweets, opt_cutoff)
    save_result(output_file, num_clusters, clustersList)
    get_sse(clustersList)

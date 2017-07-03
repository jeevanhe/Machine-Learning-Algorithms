import sys
import random
import math
from urllib.request import Request, urlopen

coordinate_points = []
centroids = []


class CoordinatePoint:
    def __init__(self, label, x, y):
        self.label = label
        self.x = x
        self.y = y


class Cluster:
    def __init__(self, points):
        # The points that belong to this cluster
        self.points = points
        # Set up the initial centroid
        self.centroid = self.getcentroid()

    def __repr__(self):
        return str(self.points)

    def update(self, points):
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.getcentroid()
        shift = get_euclidean_distance(old_centroid, self.centroid)
        return shift

    def getcentroid(self):
        numPoints = len(self.points)
        # Get a list of all coordinates in this cluster
        label = [p.label for p in self.points]
        x = [p.x for p in self.points]
        y = [p.y for p in self.points]
        # Reformat that so all x's are together, all y'z etc.
        zipped = zip(x, y)
        unzipped = zip(*zipped)

        # Calculate the mean for each dimension
        centroid_coords = [math.fsum(dList) / numPoints for dList in unzipped]
        # return the mean centroid of all points in this cluster
        return CoordinatePoint(label, centroid_coords[0], centroid_coords[1])


def get_euclidean_distance(a, b):
    ret = ((a.x - b.x) ** 2) + ((a.y - b.y) ** 2)
    return math.sqrt(ret)


def kmeans_cluster_algo(points, k, cutoff):
    iteration = 1
    # choose k random points to use as initial centroids
    initial = random.sample(points, k)
    # Create k clusters using those centroids
    clusters = [Cluster([p]) for p in initial]
    # Loop through the dataset until the clusters stabilize
    loopCounter = 0

    while (iteration <= 25):

        # Create a list of lists to hold the points in each cluster
        lists = [[] for c in clusters]
        clusterCount = len(clusters)
        loopCounter += 1
        # For every point in the dataset
        for p in points:
            smallest_distance = get_euclidean_distance(p, clusters[0].centroid)
            # Set the cluster this point belongs to
            clusterIndex = 0
            # For the remainder of the clusters
            for i in range(clusterCount - 1):
                # calculate the distance of that point to each other cluster's
                # centroid.
                distance = get_euclidean_distance(p, clusters[i + 1].centroid)
                # if the distance of the point from the centroid is lesser than
                # the previous distance, set the data point as belonging to the current cluster
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i + 1
            lists[clusterIndex].append(p)
            # Set maximumShift to zero for this iteration
        maximumShift = 0.0

        # for every cluster
        for i in range(clusterCount):
            # Calculate how far the centroid moved in this iteration
            shift = clusters[i].update(lists[i])
            # Keep track of the largest move from all cluster centroid updates
            maximumShift = max(maximumShift, shift)

        # If the centroids have stopped moving much, break from the loop
        if maximumShift < cutoff:
            print
            "Converged after %s iterations" % loopCounter
            break
        iteration += 1
    global centroids
    centroids = clusters
    return lists


def get_sse(num_clusters, list_of_clusters):
    # stores dist squared values of every point in a cluster to its centroid
    list_sse = []
    itr = 0
    total_sse = 0.0
    # for each cluster list(cluster list has a list of points belonging to individual cluster
    for i in list_of_clusters:
        totaldistance = 0.0
        for j in i:  # j refers to individual point in i
            distance = (get_euclidean_distance(centroids[itr].centroid, j) ** 2)
            totaldistance += distance
        itr += 1
        list_sse.append(totaldistance)
    total_sse = sum(list_sse)
    print ("Number of clusters - ", num_clusters)
    print ("The Validation SSE  - ", total_sse)

def save_result(out_file, num_clusters, list_of_clusters):

    file = open(out_file, 'w+')
    file.write("Cluster-ID    Points List\n")
    for i in range(num_clusters):
        file.seek(0, 2)
        str2 = str(i + 1) + "      " + str([cluster.label for cluster in list_of_clusters[i]])
        file.write(str2)
        file.write("\n")
    file.close()


if __name__ == "__main__":

    if len(sys.argv) != 4: 
        print("Invalid input!")
        print("Correct input :<num_clusters> <input-file-name> <output-file-name>")

    # get clustering value k
    num_clusters = int(sys.argv[1])
    # test file URL
    input_file_url = sys.argv[2]
    # output file
    output_file = sys.argv[3]
    
    # process the raw points into data structure
    req = Request(input_file_url, headers={'User-Agent': 'Mozilla/5.0'})
    point_data = urlopen(req).read().decode('utf-8')
    split_data = point_data.split("\r")
    all_points = []
    i = 0
    for point in split_data:
        all_points.append(point.split("\t"))
        # header skip
        if i == 0:
            i += 1
            continue
        single_point = CoordinatePoint(int(all_points[i][0]), float(all_points[i][1]), float(all_points[i][2]))
        coordinate_points.append(single_point)  
        i += 1
    opt_cutoff = 0.0
    # exit if given k value is greater than number of coordinate points
    if num_clusters > len(coordinate_points):
        print ("The number of coordinate points is ", len(coordinate_points))
        print ("Make sure num_clusters <= ", len(coordinate_points))
        exit(0)

    list_of_clusters = kmeans_cluster_algo(coordinate_points, num_clusters, opt_cutoff)
    save_result(output_file, num_clusters, list_of_clusters)
    get_sse(num_clusters, list_of_clusters)



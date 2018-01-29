#########################################################
# Author: Shaival Dalal                                 #
# Email: sd3462 [at] nyu [dot] edu                      #
# Notes: The following code has been designed to run in #
#        PySpark environment with pre-initialized Spark #
#        parameters. To use the following code for      #
#        headless spark-submit, please prepend Spark    #
#        initialization parameters.                     #
#########################################################

# Setting up environment in High Performance Computing (HPC) environment
#
## module load python/gnu/3.4.4
## export PYSPARK_PYTHON=/share/apps/python/3.4.4/bin/python
## export PYTHONHASHSEED=0
## export SPARK_YARN_USER_ENV=PYTHONHASHSEED=0
## pyspark2

# Importing basic libaries
import numpy as np
from math import sqrt
from operator import add
from pyspark.mllib.clustering import KMeans, KMeansModel

# Reading data and creating a subset of numerical columns
rawfile = sc.textFile('/sd3462/sensordatasmall/rawsensor')
rawsensor = rawfile.map(lambda line: line.split(','))
sdfilt = rawsensor.filter(lambda x:np.count_nonzero(np.array([int(x[6]), int(x[7]), int(x[8])]))>0)
sdfilt.count() # Checking the number of records
subset = sdfilt.map(lambda x: np.array([int(x[6]),int(x[7]), int(x[8])]))

# Defining errorWSSSE metric, Within Set Sum of Squared Error
def errorWSSSE(point):
	center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

# Finding optimal number of clusters, K by using WSSSE as a metric
for i in range(1,11):
	clusters = KMeans.train(subset, k=i, maxIterations=10, initializationMode="random")
    WSSSE = subset.map(lambda point: errorWSSSE(point)).reduce(add)
    print("Within Set Sum of Squared Error, k = " + str(i) + ": " + str(WSSSE))

# For k=3, we print the cluster centres
clusters = KMeans.train(subset, 3, maxIterations=10, initializationMode="random")
for i in range(0,len(clusters.centers)):
    print("Cluster " + str(i) + ": " + str(clusters.centers[i]))

# Similarly, for k=4 we print cluster centres
clusters = KMeans.train(subset, 4, maxIterations=10, initializationMode="random")
for i in range(0,len(clusters.centers)):
	print("cluster " + str(i) + ": " + str(clusters.centers[i]))

# We create function to assign a cluster number to every record by using WSSSE
def addClusterCols(x):
	point = np.array([float(x[6]), float(x[7]), float(x[8])])
    center = clusters.centers[0]
    mindist = sqrt(sum([y**2 for y in (point - center)]))
    cl = 0
    for i in range(1,len(clusters.centers)):
            center = clusters.centers[i]
            distance = sqrt(sum([y**2 for y in (point - center)]))
            if distance < mindist:
                    cl = i
                    mindist = distance
            clcenter = clusters.centers[cl]
            return (int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), float(x[5]), int(x[6]), int(x[7]), int(x[8]), int(cl), float(clcenter[0]), float(clcenter[1]), float(clcenter[2]), float(mindist))


rdd_w_clusts = sdfilt.map(lambda x: addClusterCols(x))
rdd_w_clusts.map(lambda y: (y[9],1)).reduceByKey(add).top(len(clusters.centers))

# Creating a SQL temp table called "events" in Spark for simpler querying
schema_events = sqlContext.createDataFrame(rdd_w_clusts, ('highway','sensorloc','sensorid', 'doy', 'dow', 'time','p_v','p_s','p_o', 'cluster', 'c_v','c_s', 'c_o', 'dist'))
schema_events.registerTempTable("events")

# Finding out records whose intra-cluster distance is greater than 50
sqlContext.sql("SELECT * FROM events WHERE dist>50").show()

# Displaying statistics
stats = sqlContext.sql("SELECT cluster, c_v, c_s, c_o, count(*) AS num, max(dist) AS maxdist, avg(dist) AS avgdist,stddev_pop(dist) AS stdev FROM events GROUP BY cluster, c_v, c_s, c_o ORDER BY cluster")
stats.show()

# Outlier detection
def inCluster(x, t):
    cl = x[9]
    c_v = x[10]
    c_s = x[11]
    c_o = x[12]
    distance = x[13]
    if float(distance) > float(t):
            cl = -1
            c_v = 0.0
            c_s = 0.0
            c_o = 0.0
    return (int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), float(x[5]),int(x[6]), int(x[7]), int(x[8]), int(cl), float(c_v), float(c_s), float(c_o),float(distance))

rdd_w_clusts_wnullclust = rdd_w_clusts.map(lambda x: inCluster(x,20))
rdd_w_clusts_wnullclust.map(lambda y: (y[9],1)).reduceByKey(add).top(5)

# Creating a new temp table called "event_new" to query about outliers
schema_events = sqlContext.createDataFrame(rdd_w_clusts_wnullclust, ('highway','sensorloc','sensorid', 'doy', 'dow', 'time','p_v','p_s','p_o', 'cluster', 'c_v','c_s','c_o','dist'))
schema_events.registerTempTable("events_new")

# Displaying the first 100 outliers in our "events_new" data and distance averages of our "events"  
sqlContext.sql("SELECT p_v, p_s, p_o FROM events_new WHERE cluster=-1 LIMIT 100").show(100)

sqlContext.sql("SELECT sensorid, cluster, count(*) AS num_outliers, avg(c_s) AS spdcntr, avg(dist) AS avgdist FROM events WHERE dist > 20 GROUP BY sensorid, cluster ORDER BY sensorid, cluster").show()
sqlContext.sql("SELECT cluster, doy, time, c_v,c_s,c_o, p_v,p_s,p_o FROM events WHERE cluster=0 and dist >20 ORDER BY dist").show()

# Developing KMeans model with 5 clusters and repeating the process to check for differences
clusters = KMeans.train(subset, 5, maxIterations=10, initializationMode="random")
rdd_w_clustsk5 = sdfilt.map(lambda x: addClusterCols(x))                    
schema_events = sqlContext.createDataFrame(rdd_w_clustsk5, ('highway','sensorloc','sensorid', 'doy', 'dow', 'time', 'p_v', 'p_s', 'p_o', 'cluster', 'c_v', 'c_s','c_o', 'dist'))
schema_events.registerTempTable("events_k5")

sqlContext.sql("SELECT cluster, c_v, c_s, c_o, count(*) AS num, max(dist) AS maxdist,avg(dist) AS avgdist,stddev_pop(dist) AS stdev FROM events_k5 GROUP BY cluster, c_v,c_s, c_o ORDER BY cluster").show()

rdd_w_clusts_wnullclustk5 = rdd_w_clustsk5.map(lambda x: inCluster(x,20))
rdd_w_clusts_wnullclustk5.map(lambda y: (y[9],1)).reduceByKey(add).top(5)

sqlContext.sql("SELECT sensorid, cluster, count(*) AS num_outliers, avg(c_s) AS spdcntr,avg(dist) AS avgdist FROM events_k5 WHERE dist > 20 GROUP BY sensorid, cluster ORDER BY sensorid, cluster").show()

sqlContext.sql("SELECT cluster, doy, time, c_v,c_s,c_o, p_v,p_s,p_o FROM events_k5 WHERE cluster=0 and dist >20 ORDER BY dist").show()

schema_events = sqlContext.createDataFrame(rdd_w_clusts_wnullclustk5, ('highway','sensorloc', 'sensorid', 'doy', 'dow', 'time','p_v','p_s','p_o','cluster', 'c_v','c_s','c_o','dist'))
schema_events.registerTempTable("events_new_k5")
cdata=sqlContext.sql("SELECT cluster, p_v, p_s, p_o FROM events_new_k5 ORDER BY cluster")

cdata.toPandas().to_csv("Anomalies.csv",header=True)

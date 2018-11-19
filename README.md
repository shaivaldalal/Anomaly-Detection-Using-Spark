# Anomaly Detection Using Spark
**Full Title**: Anomaly Detection for Amazon's traffic sensor data using Spark and Python

**Dependencies** (Linux HPC Environment)

| Name  	|  Version 	|
|---	|---	|
|  Spark 	| 2.2.0  	|
|  Python 	| 3.4.4  	|
|  Java 	| 1.8.0_72	  	|


**Note:** The following code has been designed to run in PySpark environment with pre-initialized Spark parameters. To use the following code for headless spark-submit, please prepend Spark initialization parameters to the code.

**Using the code**:
1. Loading modules
* `module load python/gnu/3.4.4`
* `module load java/1.8.0_72`
* `module load spark/2.2.0`

2. Setting environment variables
* `export PYSPARK_PYTHON=/share/apps/python/3.4.4/bin/python3`
* `export SPARK_YARN_USER_ENV=PYTHONHASHSEED=0`
3. Starting PySpark
* `pyspark2`

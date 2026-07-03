from pyspark.sql import SparkSession
import os

#os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"

spark = (
    SparkSession.builder
    .appName("light-cluster-test")
    .master("spark://10.55.6.41:7077")
    .config("spark.driver.host", "10.20.0.100")
    .config("spark.driver.bindAddress", "0.0.0.0")
    .getOrCreate()
)

spark.stop()
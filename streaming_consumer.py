"""
Spark Structured Streaming: consume el topic de Kafka y calcula, por ventanas de
30 segundos y departamento, la precipitacion promedio y maxima, y marca alertas.

Uso:
  spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
      src/streaming_consumer.py
"""
import argparse

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

ESQUEMA = StructType([
    StructField("estacion", StringType()),
    StructField("departamento", StringType()),
    StructField("municipio", StringType()),
    StructField("valor", DoubleType()),
    StructField("fecha_original", StringType()),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="lecturas-clima")
    ap.add_argument("--servers", default="localhost:9092")
    ap.add_argument("--umbral", type=float, default=50.0, help="mm para generar alerta")
    args = ap.parse_args()

    spark = SparkSession.builder.appName("Clima IDEAM - Streaming").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    eventos = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", args.servers)
        .option("subscribe", args.topic)
        .option("startingOffsets", "latest")
        .load()
        .select(F.from_json(F.col("value").cast("string"), ESQUEMA).alias("e"))
        .select("e.*")
        # Los datos son historicos: usamos el momento de llegada como tiempo de evento
        .withColumn("llegada", F.current_timestamp())
    )

    ventanas = (
        eventos.withWatermark("llegada", "1 minute")
        .groupBy(F.window("llegada", "30 seconds"), "departamento")
        .agg(
            F.count("*").alias("lecturas"),
            F.round(F.avg("valor"), 2).alias("precip_prom"),
            F.max("valor").alias("precip_max"),
        )
        .withColumn("alerta", F.when(F.col("precip_max") >= args.umbral, "LLUVIA FUERTE").otherwise("normal"))
    )

    consulta = (
        ventanas.writeStream.outputMode("update")
        .format("console")
        .option("truncate", False)
        .option("checkpointLocation", "checkpoint/")
        .start()
    )
    consulta.awaitTermination()


if __name__ == "__main__":
    main()

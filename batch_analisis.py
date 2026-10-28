"""
Procesamiento batch con Spark - Datos hidrometeorologicos del IDEAM.

Responde:
  1. Precipitacion mensual por departamento.
  2. Valores extremos (anomalias) por estacion (z-score > 3).
  3. Tendencia anual.
  4. Prediccion de la precipitacion del mes siguiente (regresion lineal, MLlib).

Uso:
  spark-submit src/batch_analisis.py --input data/precipitacion.csv --output output
"""
import argparse

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession, Window, functions as F


def cargar_y_limpiar(spark, ruta):
    df = spark.read.option("header", True).csv(ruta)
    # Normaliza nombres de columnas a minusculas
    df = df.toDF(*[c.strip().lower() for c in df.columns])
    print("Columnas encontradas:", df.columns)

    # NOTA: verifica que estos nombres existan en tu CSV; ajusta si es necesario.
    df = df.select(
        F.col("codigoestacion").alias("estacion"),
        F.col("departamento"),
        F.col("municipio"),
        F.col("valorobservado").cast("double").alias("valor"),
        F.coalesce(
            F.to_timestamp("fechaobservacion", "yyyy-MM-dd'T'HH:mm:ss.SSS"),
            F.to_timestamp("fechaobservacion", "MM/dd/yyyy hh:mm:ss a"),
            F.to_timestamp("fechaobservacion"),
        ).alias("fecha"),
    )

    total = df.count()
    limpio = (
        df.dropna(subset=["estacion", "departamento", "valor", "fecha"])
        .filter(F.col("valor") >= 0)
        .dropDuplicates(["estacion", "fecha"])
    )
    print(f"Registros leidos: {total:,} | registros validos: {limpio.count():,}")
    return limpio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV de precipitacion")
    ap.add_argument("--output", default="output")
    args = ap.parse_args()

    spark = SparkSession.builder.appName("Clima IDEAM - Batch").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = cargar_y_limpiar(spark, args.input)

    # Total diario por estacion
    diario = (
        df.withColumn("dia", F.to_date("fecha"))
        .groupBy("estacion", "departamento", "dia")
        .agg(F.sum("valor").alias("precip_dia"))
        .cache()
    )

    # 1. Precipitacion mensual por departamento
    mensual_est = (
        diario.withColumn("anio", F.year("dia"))
        .withColumn("mes", F.month("dia"))
        .groupBy("estacion", "departamento", "anio", "mes")
        .agg(F.sum("precip_dia").alias("precip_mes"))
    )
    mensual_dep = (
        mensual_est.groupBy("departamento", "anio", "mes")
        .agg(F.round(F.avg("precip_mes"), 2).alias("precip_mes_prom"))
        .orderBy("departamento", "anio", "mes")
    )
    mensual_dep.write.mode("overwrite").parquet(f"{args.output}/mensual_departamento")
    print("== Precipitacion mensual por departamento ==")
    mensual_dep.show(10)

    # 2. Anomalias: z-score por estacion
    w = Window.partitionBy("estacion")
    anomalias = (
        diario.withColumn("media", F.avg("precip_dia").over(w))
        .withColumn("desv", F.stddev("precip_dia").over(w))
        .withColumn("z", (F.col("precip_dia") - F.col("media")) / F.col("desv"))
        .filter(F.col("z") > 3)
        .select("estacion", "departamento", "dia", "precip_dia", F.round("z", 2).alias("z"))
        .orderBy(F.desc("z"))
    )
    anomalias.write.mode("overwrite").parquet(f"{args.output}/anomalias")
    print("== Top eventos extremos (z > 3) ==")
    anomalias.show(10)

    # 3. Tendencia anual por departamento
    anual = (
        mensual_est.groupBy("departamento", "anio")
        .agg(F.round(F.avg("precip_mes"), 2).alias("precip_mes_prom"))
        .orderBy("departamento", "anio")
    )
    anual.write.mode("overwrite").parquet(f"{args.output}/tendencia_anual")
    print("== Tendencia anual ==")
    anual.show(10)

    # 4. Prediccion del mes siguiente (serie nacional promedio)
    serie = (
        mensual_dep.groupBy("anio", "mes")
        .agg(F.avg("precip_mes_prom").alias("y"))
        .withColumn("t", F.col("anio") * 12 + F.col("mes"))
    )
    wt = Window.orderBy("t")
    serie = (
        serie.withColumn("lag1", F.lag("y", 1).over(wt))
        .withColumn("lag2", F.lag("y", 2).over(wt))
        .withColumn("lag12", F.lag("y", 12).over(wt))
        .dropna()
    )
    corte = serie.agg(F.max("anio")).first()[0] - 1  # ultimos ~2 anios para prueba
    entrenamiento = serie.filter(F.col("anio") <= corte)
    prueba = serie.filter(F.col("anio") > corte)

    ensamblador = VectorAssembler(inputCols=["lag1", "lag2", "lag12", "mes"], outputCol="features")
    modelo = LinearRegression(featuresCol="features", labelCol="y").fit(
        ensamblador.transform(entrenamiento)
    )
    pred = modelo.transform(ensamblador.transform(prueba))
    rmse = RegressionEvaluator(labelCol="y", predictionCol="prediction", metricName="rmse").evaluate(pred)
    r2 = RegressionEvaluator(labelCol="y", predictionCol="prediction", metricName="r2").evaluate(pred)
    print(f"== Modelo de regresion lineal == RMSE: {rmse:.2f} | R2: {r2:.3f}")
    pred.select("anio", "mes", "y", "prediction").orderBy("anio", "mes").show(12)

    spark.stop()


if __name__ == "__main__":
    main()

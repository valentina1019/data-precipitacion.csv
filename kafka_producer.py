"""
Productor Kafka: reenvia lecturas historicas del CSV como eventos JSON,
simulando la llegada de datos en tiempo real.

Uso:
  python src/kafka_producer.py --input data/precipitacion.csv --delay 0.2
"""
import argparse
import csv
import json
import time

from kafka import KafkaProducer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--topic", default="lecturas-clima")
    ap.add_argument("--servers", default="localhost:9092")
    ap.add_argument("--delay", type=float, default=0.2, help="segundos entre mensajes")
    ap.add_argument("--max", type=int, default=5000, help="maximo de mensajes a enviar")
    args = ap.parse_args()

    producer = KafkaProducer(
        bootstrap_servers=args.servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
    )

    enviados = 0
    with open(args.input, newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            fila = {k.strip().lower(): v for k, v in fila.items()}
            try:
                evento = {
                    "estacion": fila["codigoestacion"],
                    "departamento": fila["departamento"],
                    "municipio": fila["municipio"],
                    "valor": float(fila["valorobservado"]),
                    "fecha_original": fila["fechaobservacion"],
                }
            except (KeyError, ValueError):
                continue  # fila incompleta o con columnas distintas
            # La clave = estacion, asi cada estacion cae siempre en la misma particion
            producer.send(args.topic, key=evento["estacion"], value=evento)
            enviados += 1
            print(f"[{enviados}] {evento['departamento']} - {evento['estacion']}: {evento['valor']} mm")
            time.sleep(args.delay)
            if enviados >= args.max:
                break

    producer.flush()
    print(f"Listo: {enviados} eventos enviados al topic '{args.topic}'.")


if __name__ == "__main__":
    main()

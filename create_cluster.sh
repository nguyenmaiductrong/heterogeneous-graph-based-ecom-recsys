#!/bin/bash

CLUSTER_NAME="my-wls-pyspark-cluster"

echo "Đang tạo Dataproc cluster: $CLUSTER_NAME"

gcloud dataproc clusters create $CLUSTER_NAME \
    --master-machine-type=n2-standard-4 \
    --master-boot-disk-size=100 \
    --num-workers=3 \
    --worker-machine-type=n2-highmem-4 \
    --worker-boot-disk-size=100 \
    --image-version=2.2-debian11

echo "Khởi tạo thành công Dataproc cluster: $CLUSTER_NAME"
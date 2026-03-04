from google.cloud import storage

client = storage.Client() 
bucket = client.bucket('rhg-rees46-raw')

print(f"--- Dang kiem tra bucket: {bucket.name} ---")
blobs = list(bucket.list_blobs(max_results=5))

if not blobs:
    print("Bucket trong")
else:
    for blob in blobs:
        print(f" - {blob.name}")


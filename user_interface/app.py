import os, time
from flask import Flask, request, render_template
from google.cloud import storage
from google.cloud import dataproc_v1

app = Flask(__name__)

PROJECT_ID = os.environ.get("PROJECT_ID")
REGION     = os.environ.get("REGION", "us-central1")
BUCKET     = os.environ.get("BUCKET")
CLUSTER    = os.environ.get("CLUSTER")

JOBS = {
    "stats":  {"label": "Stats",  "script": lambda: f"gs://{BUCKET}/scripts/ui_stats.py"},
    "dt":     {"label": "Decision Tree (Regression)", "script": lambda: f"gs://{BUCKET}/scripts/ui_dt.py"},
    "kmeans": {"label": "KMeans (Clustering)", "script": lambda: f"gs://{BUCKET}/scripts/ui_kmeans.py"},
}

storage_client = storage.Client()
job_client = dataproc_v1.JobControllerClient(
    client_options={"api_endpoint": f"{REGION}-dataproc.googleapis.com:443"}
)

@app.get("/")
def home():
    default_input = f"gs://{BUCKET}/input/yellow_tripdata_2023-01.parquet"
    return render_template(
        "index.html",
        default_input=default_input,
        bucket=BUCKET,
        cluster=CLUSTER,
        region=REGION,
        jobs=JOBS,
        default_job="stats",
    )

@app.post("/run")
def run_job():
    job_key = request.form.get("job", "stats").strip()
    if job_key not in JOBS:
        return "Invalid job", 400

    input_path = request.form.get("input_path", "").strip()
    f = request.files.get("file")

    if f and f.filename:
        ts = int(time.time())
        gcs_name = f"input/uploads/{ts}_{f.filename}"
        bucket = storage_client.bucket(BUCKET)
        blob = bucket.blob(gcs_name)
        blob.upload_from_file(f.stream, content_type=f.content_type)
        input_path = f"gs://{BUCKET}/{gcs_name}"

    if not input_path.startswith("gs://"):
        return "Invalid input_path (must start with gs://)", 400

    ts = int(time.time())
    output_dir = f"gs://{BUCKET}/output/ui_{job_key}_{ts}"
    script_uri = JOBS[job_key]["script"]()

    job = {
        "placement": {"cluster_name": CLUSTER},
        "pyspark_job": {
            "main_python_file_uri": script_uri,
            "args": [input_path, output_dir],
        },
    }

    resp = job_client.submit_job(request={"project_id": PROJECT_ID, "region": REGION, "job": job})
    job_id = resp.reference.job_id

    job_url = f"https://console.cloud.google.com/dataproc/jobs/{job_id}?region={REGION}&project={PROJECT_ID}"
    out_url = f"https://console.cloud.google.com/storage/browser/{BUCKET}/output/ui_{job_key}_{ts}"

    return render_template(
        "done.html",
        job_id=job_id,
        job_name=JOBS[job_key]["label"],
        input_path=input_path,
        output_dir=output_dir,
        job_url=job_url,
        out_url=out_url,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))

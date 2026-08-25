import kagglehub

# downloads the nasa c-mapss dataset from kaggle to a local cache.
# requires a kaggle api token at ~/.kaggle/kaggle.json — see README for setup.
# only needs to run once; kagglehub skips the download on subsequent calls if
# the files are already cached locally.
path = kagglehub.dataset_download("behrad3d/nasa-cmaps")

# the printed path is where the raw .txt files now live — the pipeline reads from there
print("path to dataset files:", path)

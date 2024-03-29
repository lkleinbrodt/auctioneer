import boto3
import os
from config import *
from io import BytesIO, StringIO
import json
import pandas as pd
import yaml
import torch
import os
import zipfile
import boto3

class S3Client:
    def __init__(self):
        self.bucket = 'auctioneer1'
        try:
            self.s3 = boto3.client(
                "s3",
                aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
                aws_secret_access_key=os.environ["AWS_SECRET_KEY"],
            )
            
        except KeyError:
            raise ValueError("AWS credentials not found. Please set AWS_ACCESS_KEY and AWS_SECRET_KEY environment variables.")
        
    def write_csv(self, df, path, index = True):
        buffer = BytesIO()
        df.to_csv(buffer, index = index)
        self.s3.put_object(Body=buffer.getvalue(), Bucket=self.bucket, Key=path)
        return True
    
    def read_csv(self, path):
        s3_object = self.s3.get_object(Bucket=self.bucket, Key=path)
        contents = s3_object["Body"].read()
        df = pd.read_csv(BytesIO(contents))
        return df

    def read_parquet(self, path):
        s3_object = self.s3.get_object(Bucket=self.bucket, Key=path)
        contents = s3_object["Body"].read()
        df = pd.read_parquet(BytesIO(contents))
        return df
    
    def get_all_objects(self, **base_kwargs):
        continuation_token = None
        while True:
            list_kwargs = dict(MaxKeys=1000, **base_kwargs)
            if continuation_token:
                list_kwargs["ContinuationToken"] = continuation_token
            response = self.s3.list_objects_v2(**list_kwargs)
            yield from response.get("Contents", [])
            if not response.get("IsTruncated"):  # At the end of the list?
                break
            continuation_token = response.get("NextContinuationToken")
            
    def download_file(self, s3_path, local_path):
        self.s3.download_file(Bucket = self.bucket, Key = s3_path, Filename=local_path)
        return True
    
    def upload_file(self, local_path, s3_path):
        self.s3.upload_file(Bucket = self.bucket, Key = s3_path, Filename = local_path)
        return True
    
    def save_json(self, data, path):
        self.s3.put_object(Body = json.dumps(data), Bucket=self.bucket, Key = path)
    
    def load_json(self, path):
        s3_object = self.s3.get_object(Bucket=self.bucket, Key=path)
        contents = s3_object["Body"].read()
        return json.loads(contents)

    def save_yaml(self, obj, path):
        buffer = StringIO()
        yaml.safe_dump(obj, buffer)
        self.s3.put_object(Body=buffer.getvalue(), Bucket=self.bucket, Key=path)
        return True
    
    def save_model(self, model, path):
        model_state_dict = model.state_dict()
        buffer = BytesIO()
        torch.save(model_state_dict, buffer)
        buffer.seek(0)
        self.s3.put_object(Body=buffer.getvalue(), Bucket=self.bucket, Key=path)
        
        self.save_json(model.startup_params, path.replace('.pt', '_startup_params.json'))
        
        return True

    def upload_compressed_directory(self, local_path, s3_path):
        assert os.path.isdir(local_path), f"{local_path} is not a directory"
        assert s3_path.endswith('.zip'), f"{s3_path} must end with .zip"
        # Create a temporary zip file
        zip_file_path = "/tmp/compressed.zip"
        with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, local_path))
                    
        self.upload_file(zip_file_path, s3_path)

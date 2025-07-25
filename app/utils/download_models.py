import os
import zipfile
import logging
from minio import Minio
from tqdm import tqdm
from app.config import MinioConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MinioConnector:

    minio_config = MinioConfig()

    MINIO_URL = minio_config.minio_url
    MINIO_USERNAME = minio_config.minio_username
    MINIO_PASSWORD = minio_config.minio_password
    MINIO_BUCKET_NAME = minio_config.minio_bucket_name
    SECURE = minio_config.secure

    def __init__(self):
        try:
            logger.info(f"Attempting to connect to MinIO at {self.MINIO_URL}...")
            self.client = Minio(self.MINIO_URL,
                                self.MINIO_USERNAME,
                                self.MINIO_PASSWORD,
                                secure=self.SECURE)
        except Exception as e:
            logger.error(f"Failed to establish connection with MinIO: {str(e)}")
            raise

    def download_to_local(self, minio_source_path, minio_download_dir, model_path):
        try:
            os.makedirs(minio_download_dir, exist_ok=True)
            logger.info(f"Created directory: {minio_download_dir}")
            
            # Check if bucket exists
            if not self.client.bucket_exists(self.MINIO_BUCKET_NAME):
                raise Exception(f"Bucket {self.MINIO_BUCKET_NAME} does not exist")
            
            logger.info(f"Downloading {minio_source_path} from MinIO bucket {self.MINIO_BUCKET_NAME}")
            
            file_stat = self.client.stat_object(self.MINIO_BUCKET_NAME, minio_source_path)
            total_size = file_stat.size
            destination_zip_path = os.path.join(minio_download_dir, os.path.basename(minio_source_path))
            logger.info(f"Destination zip path: {destination_zip_path}")

            # Get object data with progress tracking
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                try:
                    data = self.client.get_object(self.MINIO_BUCKET_NAME, minio_source_path)
                    with open(destination_zip_path, 'wb') as file_data:
                        for chunk in data.stream(32*1024):
                            file_data.write(chunk)
                            pbar.update(len(chunk))
                except Exception as e:
                    raise Exception(f"Download failed: {str(e)}")

            logger.info(f"Download completed. Extracting to {minio_download_dir}")

            # Extract with progress tracking
            with zipfile.ZipFile(destination_zip_path, 'r') as zip_ref:
                zip_members = zip_ref.namelist()
                with tqdm(total=len(zip_members), desc="Extracting") as pbar:
                    for member in zip_members:
                        zip_ref.extract(member, minio_download_dir)
                        pbar.update(1)
            
            logger.info(f"Extraction completed. Cleaning up temporary files.")
            os.remove(destination_zip_path)

            logger.info(f"Successfully downloaded and extracted model to {minio_download_dir}")
            
        except Exception as e:
            logger.error(f"Error during download/extraction: {str(e)}")
            raise

def download_model_if_not_exists(paths):
    for model_type, model_info in paths.items():
        model_path = model_info['model_path']
        minio_source_path = model_info['minio_source_path']
        minio_download_dir = model_info['minio_download_dir']

        if not os.path.exists(model_path):
            print(f"Downloading Model from MinIO: {minio_source_path}")
            MinioConnector().download_to_local(minio_source_path, minio_download_dir, model_path)

def download_all_models_from_config():
    minio_config = MinioConfig()

    paths = {
        'deduplication': {
            'model_path': os.path.join(minio_config.download_path, 'deduplication', 'models'),
            'minio_source_path': minio_config.model_object_path,
            'minio_download_dir': minio_config.download_path,
        }
    }
    download_model_if_not_exists(paths)

## Helper Functions:
def test_minio_connection():
    try:
        connector = MinioConnector()
        # Test by listing buckets.
        buckets = list(connector.client.list_buckets())
        print("✅ MinIO connection successful")
        return True
    except Exception as e:
        print(f"❌ MinIO connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        if test_minio_connection():
            download_all_models_from_config()
        else:
            logger.error("MinIO connection test failed. Aborting download.")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
import unittest
import requests
import os
import time
import threading
from app import app
import json

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start Flask app in testing mode
        app.config['TESTING'] = True
        cls.client = app.test_client()
        cls.base_url = 'http://localhost:5010'

    def test_predict_no_file(self):
        """Test prediction endpoint without file"""
        response = self.client.post('/web/predict')
        self.assertEqual(response.status_code, 400)
        self.assertIn('No file uploaded', response.json['error'])

    def test_predict_invalid_file_type(self):
        """Test prediction endpoint with invalid file type"""
        data = {
            'file': (open(__file__, 'rb'), 'test.py')
        }
        response = self.client.post('/web/predict', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn('Invalid file', response.json['error'])

    def test_download_invalid_token(self):
        """Test download endpoint with invalid token"""
        response = self.client.get('/download/invalid_token')
        self.assertEqual(response.status_code, 401)
        self.assertIn('Invalid or expired token', response.json['error'])

def make_api_request(test_file, params, results, index, endpoint='web'):
    """Helper function to make API request and test download"""
    url = f'http://localhost:5010/{endpoint}/predict'
    
    try:
        with open(test_file, 'rb') as f:
            files = {
                'file': ('test.zip', f, 'application/zip')
            }
            # Make prediction request
            response = requests.post(url, data=params, files=files)
            
            if endpoint == 'web':
                if response.status_code == 200:
                    task_id = response.json()['task_id']
                    # Poll for completion
                    status_url = f'http://localhost:5010/web/status/{task_id}'
                    while True:
                        status_response = requests.get(status_url)
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            if status_data['status'] == 'completed':
                                # Test download with token
                                download_url = f'http://localhost:5010/download/{status_data["download_token"]}'
                                download_response = requests.get(download_url)
                                
                                results[index] = {
                                    'predict_status': response.status_code,
                                    'task_status': status_data['status'],
                                    'progress': status_data['progress'],
                                    'download_status': download_response.status_code,
                                    'download_success': download_response.status_code == 200
                                }
                                break
                            elif status_data['status'] == 'failed':
                                results[index] = {
                                    'error': 'Task processing failed'
                                }
                                break
                        time.sleep(1)  # Poll every second
                else:
                    results[index] = {
                        'predict_status': response.status_code,
                        'error': response.json().get('error', 'Unknown error')
                    }
            else:  # Direct API
                results[index] = {
                    'predict_status': response.status_code,
                    'download_success': response.status_code == 200 and response.headers.get('content-type') == 'application/zip'
                }

    except Exception as e:
        results[index] = {
            'error': str(e)
        }

def test_concurrent_requests(endpoint='web'):
    """
    Test multiple concurrent API requests
    """
    # Example parameters
    params = {
        'input_type': '0',
        'classification_threshold': '0.35',
        'prediction_threshold': '0.5',
        'save_labeled_image': 'false',
        'output_type': '0',
        'yolo_model_type': 'n'
    }

    # Check if test.zip exists
    test_file = 'test_data/test.zip'
    if not os.path.exists(test_file):
        print(f"Please place a test ZIP file at {test_file}")
        return

    # Number of concurrent requests
    num_requests = 12  # Test queue overflow
    threads = []
    results = [None] * num_requests

    print(f"\nTesting {num_requests} concurrent requests on /{endpoint}/predict...")

    # Start concurrent requests
    for i in range(num_requests):
        thread = threading.Thread(
            target=make_api_request,
            args=(test_file, params, results, i, endpoint)
        )
        threads.append(thread)
        thread.start()

    # Wait for all requests to complete
    for thread in threads:
        thread.join()

    # Check results
    successful_requests = 0
    successful_downloads = 0
    queue_full_responses = 0
    
    for i, result in enumerate(results):
        print(f"\nRequest {i + 1} result:")
        if 'error' in result:
            print(f"Error: {result['error']}")
            if 'Server is busy' in str(result['error']):
                queue_full_responses += 1
        else:
            print(f"Prediction Status: {result['predict_status']}")
            if endpoint == 'web':
                if 'task_status' in result:
                    print(f"Task Status: {result['task_status']}")
                    print(f"Progress: {result['progress']}%")
            if result['predict_status'] == 200:
                successful_requests += 1
            if result.get('download_success'):
                successful_downloads += 1

    print(f"\nSuccessful predictions: {successful_requests}/{num_requests}")
    print(f"Successful downloads: {successful_downloads}/{num_requests}")
    if endpoint == 'web':
        print(f"Queue full responses: {queue_full_responses}")

def test_file_cleanup():
    """
    Test if files are cleaned up after MAX_FILE_AGE_HOURS
    """
    print("\nTesting file cleanup...")
    print("Waiting for 2 hours and 5 minutes to test cleanup...")
    time.sleep(7500)  # 2 hours and 5 minutes
    
    # Check if files are cleaned up
    output_files = os.listdir('run/output')
    input_files = os.listdir('input')
    
    print(f"Remaining output files: {len(output_files)}")
    print(f"Remaining input files: {len(input_files)}")

if __name__ == '__main__':
    # Run unit tests
    unittest.main(argv=[''], exit=False)

    print("\nTesting web API with queuing...")
    test_concurrent_requests('web')

    print("\nTesting direct API...")
    test_concurrent_requests('api')
    
    # Uncomment to test file cleanup (takes 2+ hours)
    # test_file_cleanup() 
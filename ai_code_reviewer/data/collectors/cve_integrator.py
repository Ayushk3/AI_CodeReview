# ai_code_reviewer/data/collectors/cve_integrator.py

import requests
import json
import os
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Iterator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CVEIntegrator:
    """
    Integrates with the National Vulnerability Database (NVD) API to collect CVE information.
    """

    NVD_API_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0/"
    DEFAULT_PAGE_SIZE = 2000 # Max results per page for NVD API 2.0 is often 2000

    def __init__(self, api_key: Optional[str] = None, output_dir: str = "cve_data"):
        """
        Initializes the CVEIntegrator.

        Args:
            api_key (Optional[str]): Your NVD API key for higher rate limits.
                                      Register at https://nvd.nist.gov/developers/request-an-api-key.
            output_dir (str): Directory to save collected CVE data.
        """
        self.api_key = api_key
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.headers = {"apiKey": api_key} if api_key else {}
        logger.info(f"CVEIntegrator initialized. CVE data will be saved to: {self.output_dir}")
        if not api_key:
            logger.warning("No NVD API key provided. Rate limits will be strict (1 request/min). "
                           "Consider obtaining an API key from https://nvd.nist.gov/developers/request-an-api-key "
                           "for 100 requests/min.")

    def _make_api_request(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Makes a GET request to the NVD API with exponential backoff for rate limiting.

        Args:
            params (Dict[str, Any]): Dictionary of query parameters for the API.

        Returns:
            Optional[Dict[str, Any]]: JSON response as a dictionary, or None on failure.
        """
        retries = 5
        for i in range(retries):
            try:
                response = requests.get(self.NVD_API_BASE_URL, params=params, headers=self.headers, timeout=30)
                response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
                return response.json()
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                if status_code == 403: # Forbidden, often due to API key issues or rate limit
                    logger.error(f"API Key issue or IP blocked. Status 403: {e}. Check your API key and IP address.")
                    return None
                elif status_code == 429: # Too Many Requests
                    wait_time = 2 ** i * 60 # Exponential backoff with initial 1 min for 429
                    logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retrying ({i+1}/{retries})...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"HTTP error {status_code} during NVD API request: {e}. Params: {params}")
                    return None
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error during NVD API request: {e}")
                time.sleep(5) # Short sleep for connection errors
            except requests.exceptions.Timeout:
                logger.warning(f"NVD API request timed out. Retrying ({i+1}/{retries})...")
                time.sleep(10) # Longer sleep for timeouts
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON response from NVD API: {e}. Response: {response.text}")
                return None
            except Exception as e:
                logger.error(f"An unexpected error occurred during NVD API request: {e}. Params: {params}")
                return None
        logger.error(f"Failed to fetch data from NVD API after {retries} retries.")
        return None

    def collect_cves_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        output_filename: str = "cves_by_date.json",
        max_cves: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Collects CVEs modified within a specified date range from NVD.
        NVD API has a maximum date range of 120 days. For longer periods, this method
        will automatically chunk the requests.

        Args:
            start_date (datetime): The start date (inclusive).
            end_date (datetime): The end date (inclusive).
            output_filename (str): Name of the file to save the collected CVEs.
            max_cves (Optional[int]): Maximum number of CVEs to collect.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a CVE record.
        """
        all_cves: List[Dict[str, Any]] = []
        current_start_date = start_date
        # NVD API has a 120-day limit for modified dates
        MAX_DATE_RANGE = timedelta(days=115) # Using 115 to be safe

        while current_start_date <= end_date:
            segment_end_date = min(current_start_date + MAX_DATE_RANGE, end_date)
            logger.info(f"Collecting CVEs from {current_start_date.isoformat()} to {segment_end_date.isoformat()}...")

            total_results = -1
            start_index = 0
            retrieved_count = 0

            while total_results == -1 or start_index < total_results:
                params = {
                    "lastModStartDate": current_start_date.isoformat() + "T00:00:00.000",
                    "lastModEndDate": segment_end_date.isoformat() + "T23:59:59.999",
                    "resultsPerPage": self.DEFAULT_PAGE_SIZE,
                    "startIndex": start_index
                }

                response_data = self._make_api_request(params)
                if response_data is None:
                    logger.error(f"Failed to fetch data for date range {current_start_date} to {segment_end_date}. Stopping for this segment.")
                    break

                if total_results == -1: # First call for this segment
                    total_results = response_data.get("totalResults", 0)
                    logger.info(f"Found {total_results} CVEs in the range {current_start_date.date()} to {segment_end_date.date()}.")
                
                vulnerabilities = response_data.get("vulnerabilities", [])
                if not vulnerabilities:
                    logger.info("No more vulnerabilities in this segment/page.")
                    break

                for vuln in vulnerabilities:
                    all_cves.append(vuln.get("cve", {})) # Extract the 'cve' object
                    retrieved_count += 1
                    if max_cves and len(all_cves) >= max_cves:
                        logger.info(f"Reached max_cves ({max_cves}). Stopping collection.")
                        self._save_cves(all_cves, output_filename)
                        return all_cves

                start_index += len(vulnerabilities)
                # Introduce a small delay to respect rate limits even with API key
                # NVD API is 100 requests/min with key, 1 request/min without.
                # If no key, 60 seconds. With key, 60/100 = 0.6 seconds.
                time.sleep(0.7 if self.api_key else 60)

            current_start_date = segment_end_date + timedelta(days=1)
        
        self._save_cves(all_cves, output_filename)
        return all_cves

    def _save_cves(self, cves: List[Dict[str, Any]], filename: str):
        """
        Saves a list of CVE dictionaries to a JSON file.
        """
        filepath = os.path.join(self.output_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(cves, f, indent=4)
            logger.info(f"Successfully saved {len(cves)} CVEs to {filepath}")
        except Exception as e:
            logger.error(f"Error saving CVEs to {filepath}: {e}")

    def load_cves(self, filename: str) -> List[Dict[str, Any]]:
        """
        Loads CVE data from a JSON file.

        Args:
            filename (str): The name of the file to load.

        Returns:
            List[Dict[str, Any]]: A list of CVE dictionaries.
        """
        filepath = os.path.join(self.output_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    cves = json.load(f)
                logger.info(f"Loaded {len(cves)} CVEs from {filepath}")
                return cves
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {filepath}: {e}")
                return []
            except Exception as e:
                logger.error(f"Error loading CVEs from {filepath}: {e}")
                return []
        else:
            logger.warning(f"CVE file not found: {filepath}")
            return []

# --- Unit Tests ---
def run_unit_tests(api_key: Optional[str] = None):
    """
    Runs unit tests for the CVEIntegrator class.
    Note: These tests require network access to NVD API and might hit rate limits
    if no API key is provided or too many tests are run.
    """
    logger.info("Running unit tests for CVEIntegrator...")
    temp_output_dir = "test_cve_data"
    
    # Clean up any previous test directory
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)

    # Test 1: Initialization
    try:
        integrator = CVEIntegrator(output_dir=temp_output_dir)
        assert os.path.exists(temp_output_dir)
        logger.info("Test 1 (Initialization): PASSED")
    except AssertionError:
        logger.error("Test 1 (Initialization): FAILED")
    
    # Test 2: Basic API Request (minimal, should hit rate limit without key quickly)
    integrator = CVEIntegrator(api_key=api_key, output_dir=temp_output_dir)
    test_start_date = datetime.now() - timedelta(days=5)
    test_end_date = datetime.now()
    try:
        # Request a very small number of CVEs to avoid heavy downloads
        # This tests if the API call and basic parsing work
        logger.info("Test 2 (Basic API Request - limited): Attempting to fetch 1-2 CVEs.")
        cves_collected = integrator.collect_cves_by_date_range(
            test_start_date, test_end_date, "test_cves_small.json", max_cves=2
        )
        assert len(cves_collected) > 0, "Test 2: Failed to collect any CVEs."
        assert "id" in cves_collected[0] and "descriptions" in cves_collected[0], "Test 2: CVE structure incomplete."
        logger.info(f"Test 2 (Basic API Request): PASSED (collected {len(cves_collected)} CVEs)")
    except Exception as e:
        logger.error(f"Test 2 (Basic API Request): FAILED with exception {e}")

    # Test 3: Load CVEs from file
    try:
        loaded_cves = integrator.load_cves("test_cves_small.json")
        assert len(loaded_cves) == len(cves_collected), "Test 3: Loaded CVE count mismatch."
        logger.info("Test 3 (Load CVEs): PASSED")
    except AssertionError:
        logger.error("Test 3 (Load CVEs): FAILED")
    except Exception as e:
        logger.error(f"Test 3 (Load CVEs): FAILED with exception {e}")

    # Clean up test directory
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)
    logger.info("CVEIntegrator unit tests completed.")

# --- Performance Profiling (Conceptual) ---
def profile_cve_integrator_performance(api_key: Optional[str] = None):
    """
    Profiles the performance of the CVEIntegrator.
    This will involve making actual API calls.
    """
    logger.info("\nStarting CVEIntegrator performance profiling...")
    profiling_integrator = CVEIntegrator(api_key=api_key, output_dir="temp_cve_profiling")
    
    # Clean up previous profiling run
    if os.path.exists("temp_cve_profiling"):
        shutil.rmtree("temp_cve_profiling")

    # Profile collecting CVEs for a specific recent period
    start_date = datetime.now() - timedelta(days=30) # Last 30 days
    end_date = datetime.now()

    start_time = datetime.now()
    collected_cves = []
    try:
        # Collect a limited number of CVEs to prevent extremely long run times or excessive API calls
        collected_cves = profiling_integrator.collect_cves_by_date_range(
            start_date, end_date, "profiling_cves.json", max_cves=500
        )
    except Exception as e:
        logger.error(f"Error during CVE integrator profiling: {e}")
    finally:
        # Clean up the downloaded CVEs
        if os.path.exists("temp_cve_profiling"):
            shutil.rmtree("temp_cve_profiling")

    end_time = datetime.now()
    duration = end_time - start_time

    logger.info(f"Profiling complete for CVEIntegrator:")
    logger.info(f"  CVEs collected: {len(collected_cves)}")
    logger.info(f"  Total duration: {duration}")
    if len(collected_cves) > 0:
        logger.info(f"  Average time per CVE (approx): {duration / len(collected_cves)}")
    else:
        logger.info("  No CVEs collected for average time calculation.")


# --- Usage Example ---
if __name__ == "__main__":
    import shutil

    # Replace with your actual NVD API Key if you have one
    # It's recommended to store this in an environment variable or a config file, not directly in code.
    # For demonstration, you can put it here, but remember to remove it for public repos.
    NVD_API_KEY = os.getenv("NVD_API_KEY") # Get from environment variable

    # Run unit tests
    # Consider running with a valid API key if available for full testing
    run_unit_tests(api_key=NVD_API_KEY)

    # --- Data Collection Example ---
    logger.info("\n--- Demonstrating CVE Data Collection ---")
    cve_integrator = CVEIntegrator(api_key=NVD_API_KEY, output_dir="collected_cve_data")

    # Collect CVEs from the last 6 months (adjust as needed, will be chunked by the script)
    six_months_ago = datetime.now() - timedelta(days=180)
    today = datetime.now()

    collected_cves_data = cve_integrator.collect_cves_by_date_range(
        start_date=six_months_ago,
        end_date=today,
        output_filename="recent_cves.json",
        max_cves=1000 # Limit for demonstration purposes
    )

    logger.info(f"\n--- CVE Collection Summary ---")
    logger.info(f"Total CVEs collected: {len(collected_cves_data)}")

    # Example of loading collected CVEs
    loaded_cves_example = cve_integrator.load_cves("recent_cves.json")
    if loaded_cves_example:
        logger.info(f"First loaded CVE ID: {loaded_cves_example[0].get('id')}")
        logger.info(f"Description of first CVE: {loaded_cves_example[0].get('descriptions', [{'value': 'N/A'}])[0].get('value')}")
    
    # --- Performance Profiling ---
    profile_cve_integrator_performance(api_key=NVD_API_KEY)

    # Clean up the demonstration output directory
    if os.path.exists("collected_cve_data"):
        shutil.rmtree("collected_cve_data")
    if os.path.exists("temp_cve_profiling"):
        shutil.rmtree("temp_cve_profiling")
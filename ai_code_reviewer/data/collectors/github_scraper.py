# ai_code_reviewer/data/collectors/github_scraper.py

import os
import shutil
import logging
from typing import Iterator, Tuple, Dict, Any, List
import pygit2
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitHubScraper:
    """
    Scrapes GitHub repositories for Python/Java code samples, focusing on bug-fix commits.
    Utilizes pygit2 for efficient Git operations.
    """

    def __init__(self, output_dir: str = "scraped_repos"):
        """
        Initializes the GitHubScraper.

        Args:
            output_dir (str): Directory to clone repositories into.
        """
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"GitHubScraper initialized. Repositories will be cloned to: {self.output_dir}")

    def _clone_repository(self, repo_url: str, repo_path: str) -> pygit2.Repository:
        """
        Clones a Git repository.

        Args:
            repo_url (str): The URL of the repository to clone.
            repo_path (str): The local path where the repository will be cloned.

        Returns:
            pygit2.Repository: The cloned repository object.

        Raises:
            pygit2.errors.GitError: If cloning fails.
        """
        if os.path.exists(repo_path):
            logger.info(f"Repository already exists at {repo_path}. Opening existing repository.")
            try:
                return pygit2.Repository(repo_path)
            except pygit2.errors.GitError as e:
                logger.error(f"Failed to open existing repository at {repo_path}: {e}")
                raise
        else:
            logger.info(f"Cloning {repo_url} to {repo_path}...")
            try:
                repo = pygit2.clone_repository(repo_url, repo_path)
                logger.info(f"Successfully cloned {repo_url}")
                return repo
            except pygit2.errors.GitError as e:
                logger.error(f"Failed to clone {repo_url}: {e}")
                raise

    def _is_bug_fix_commit(self, commit: pygit2.Commit) -> bool:
        """
        Heuristically determines if a commit is a bug-fix commit.
        Checks for keywords like "fix", "bug", "issue", "patch".

        Args:
            commit (pygit2.Commit): The commit object.

        Returns:
            bool: True if it's likely a bug-fix commit, False otherwise.
        """
        message = commit.message.lower()
        bug_keywords = ["fix", "bug", "issue", "defect", "patch", "error"]
        return any(keyword in message for keyword in bug_keywords)

    def _extract_file_content(self, tree: pygit2.Tree, filepath: str) -> str | None:
        """
        Extracts the content of a file from a Git tree.

        Args:
            tree (pygit2.Tree): The tree object.
            filepath (str): The path to the file within the tree.

        Returns:
            str | None: The content of the file as a string, or None if the file is not found
                        or cannot be decoded.
        """
        try:
            entry = tree[filepath]
            blob = tree.repository[entry.oid]
            return blob.data.decode('utf-8', errors='ignore')
        except KeyError:
            logger.debug(f"File '{filepath}' not found in tree.")
            return None
        except UnicodeDecodeError:
            logger.warning(f"Could not decode file '{filepath}' content in commit.")
            return None
        except Exception as e:
            logger.error(f"Error extracting content for '{filepath}': {e}")
            return None

    def find_bug_fix_pairs(
        self,
        repo_url: str,
        target_languages: List[str],
        max_commits: int = 100,
        since_date: datetime | None = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Finds and yields bug-fix commit "before" and "after" code pairs for specified languages.

        Args:
            repo_url (str): The URL of the GitHub repository.
            target_languages (List[str]): List of programming languages to filter by (e.g., ["python", "java"]).
            max_commits (int): Maximum number of commits to process.
            since_date (datetime | None): Only process commits newer than this date.

        Yields:
            Iterator[Dict[str, Any]]: A dictionary for each bug-fix pair found,
                                      containing 'repo_url', 'commit_hash', 'filepath',
                                      'before_code', 'after_code', 'commit_message', 'language'.
        """
        repo_name = repo_url.split('/')[-1].replace(".git", "")
        repo_path = os.path.join(self.output_dir, repo_name)

        try:
            repo = self._clone_repository(repo_url, repo_path)
        except pygit2.errors.GitError:
            logger.error(f"Skipping repository {repo_url} due to cloning error.")
            return

        head = repo.head
        if head is None:
            logger.warning(f"Repository {repo_url} has no HEAD. Skipping.")
            return

        logger.info(f"Scanning commits in {repo_url}...")
        commit_count = 0
        for commit in repo.walk(head.target, pygit2.GIT_SORT_TIME):
            if commit_count >= max_commits:
                logger.info(f"Reached max_commits ({max_commits}) for {repo_url}.")
                break

            if since_date and commit.commit_time < since_date.timestamp():
                logger.debug(f"Skipping commit {commit.hex} as it's older than {since_date}.")
                continue

            if not self._is_bug_fix_commit(commit):
                logger.debug(f"Commit {commit.hex} is not a bug-fix commit. Skipping.")
                continue

            # Skip merge commits as they often don't represent a single logical change
            if len(commit.parents) != 1:
                logger.debug(f"Skipping merge commit {commit.hex}.")
                continue

            parent_commit = commit.parents[0]
            diff = repo.diff(parent_commit.tree, commit.tree)

            for patch in diff:
                delta = patch.delta
                old_file_path = delta.old_file.path
                new_file_path = delta.new_file.path

                # Ensure it's a modification, not an addition/deletion of a file
                if delta.status != pygit2.GIT_DELTA_MODIFIED:
                    continue

                # Determine language based on file extension
                file_extension = os.path.splitext(new_file_path)[1].lower()
                language = None
                if file_extension == ".py" and "python" in [lang.lower() for lang in target_languages]:
                    language = "python"
                elif file_extension == ".java" and "java" in [lang.lower() for lang in target_languages]:
                    language = "java"

                if language is None:
                    continue

                before_code = self._extract_file_content(parent_commit.tree, old_file_path)
                after_code = self._extract_file_content(commit.tree, new_file_path)

                if before_code is None or after_code is None:
                    logger.debug(f"Could not extract both before/after code for {new_file_path} in commit {commit.hex}.")
                    continue

                yield {
                    "repo_url": repo_url,
                    "commit_hash": commit.hex,
                    "filepath": new_file_path,
                    "before_code": before_code,
                    "after_code": after_code,
                    "commit_message": commit.message,
                    "language": language,
                    "commit_time": datetime.fromtimestamp(commit.commit_time).isoformat()
                }
                commit_count += 1

    def clean_up(self, repo_url: str):
        """
        Removes a cloned repository from the local disk.

        Args:
            repo_url (str): The URL of the repository to clean up.
        """
        repo_name = repo_url.split('/')[-1].replace(".git", "")
        repo_path = os.path.join(self.output_dir, repo_name)
        if os.path.exists(repo_path):
            try:
                shutil.rmtree(repo_path)
                logger.info(f"Cleaned up repository: {repo_path}")
            except OSError as e:
                logger.error(f"Error removing directory {repo_path}: {e}")
        else:
            logger.info(f"Repository path {repo_path} does not exist. Nothing to clean.")

def profile_scraper_performance(repo_url: str, target_languages: List[str], max_commits: int):
    """
    Profiles the performance of the GitHubScraper.

    Args:
        repo_url (str): The URL of the GitHub repository.
        target_languages (List[str]): Languages to scrape.
        max_commits (int): Maximum commits to process.
    """
    logger.info(f"Starting performance profiling for {repo_url} with max_commits={max_commits}")
    scraper = GitHubScraper(output_dir="temp_scraped_repos_profiling")
    start_time = datetime.now()
    bug_fix_count = 0
    try:
        for _ in scraper.find_bug_fix_pairs(repo_url, target_languages, max_commits):
            bug_fix_count += 1
    except Exception as e:
        logger.error(f"Error during profiling: {e}")
    finally:
        scraper.clean_up(repo_url) # Ensure cleanup after profiling
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Profiling complete for {repo_url}:")
    logger.info(f"  Total bug-fix pairs found: {bug_fix_count}")
    logger.info(f"  Total duration: {duration}")
    logger.info(f"  Average time per pair (approx): {duration / max(1, bug_fix_count)}")

# --- Unit Tests (Conceptual, as pygit2 requires actual repo setup) ---
def run_unit_tests():
    """
    Conceptual unit tests for GitHubScraper.
    Note: Real tests would require mock Git repositories or a temporary,
    controlled Git environment.
    """
    logger.info("Running conceptual unit tests for GitHubScraper...")

    # Test 1: Initialization
    try:
        scraper = GitHubScraper(output_dir="test_output")
        assert os.path.exists("test_output")
        logger.info("Test 1 (Initialization): PASSED")
    except AssertionError:
        logger.error("Test 1 (Initialization): FAILED")
    except Exception as e:
        logger.error(f"Test 1 (Initialization): FAILED with exception {e}")
    finally:
        if os.path.exists("test_output"):
            shutil.rmtree("test_output") # Clean up test directory

    # Test 2: Cloning (simulated)
    # In a real scenario, you'd mock pygit2.clone_repository
    # For now, just check the call structure.
    logger.info("Test 2 (Cloning): Requires manual verification or mocking.")
    # Example:
    # from unittest.mock import patch, MagicMock
    # with patch('pygit2.clone_repository') as mock_clone:
    #     mock_clone.return_value = MagicMock(spec=pygit2.Repository)
    #     scraper = GitHubScraper("temp_test_clone")
    #     repo = scraper._clone_repository("http://example.com/repo.git", "temp_test_clone/repo")
    #     mock_clone.assert_called_once_with("http://example.com/repo.git", "temp_test_clone/repo")
    #     shutil.rmtree("temp_test_clone")

    # Test 3: _is_bug_fix_commit (simulated)
    class MockCommit:
        def __init__(self, message: str):
            self.message = message
            self.hex = "mock_hash"
            self.commit_time = datetime.now().timestamp()
            self.parents = [self] # For diffing, a mock parent

    scraper = GitHubScraper("temp_test_bug_fix")
    assert scraper._is_bug_fix_commit(MockCommit("Fix: A bug was squashed.")) == True
    assert scraper._is_bug_fix_commit(MockCommit("feat: New feature added")) == False
    logger.info("Test 3 (_is_bug_fix_commit): PASSED (conceptual)")
    if os.path.exists("temp_test_bug_fix"):
        shutil.rmtree("temp_test_bug_fix")

    # Test 4: clean_up (simulated)
    test_repo_dir = os.path.join(scraper.output_dir, "test_repo_to_delete")
    os.makedirs(test_repo_dir, exist_ok=True)
    scraper.clean_up("http://example.com/test_repo_to_delete.git")
    assert not os.path.exists(test_repo_dir)
    logger.info("Test 4 (clean_up): PASSED (conceptual)")
    if os.path.exists(scraper.output_dir):
        shutil.rmtree(scraper.output_dir)


# --- Usage Example ---
if __name__ == "__main__":
    run_unit_tests()

    # --- Sample Dataset Generation ---
    # Due to the nature of pygit2 requiring actual git repositories,
    # and the potential for long runtimes and large downloads,
    # this part will focus on demonstrating usage with a few *example*
    # public repositories known to have some history.

    # IMPORTANT: Replace with actual URLs of small, public repositories for testing.
    # Avoid very large repositories for initial testing to manage download times.
    sample_repositories = [
        "https://github.com/pallets/flask.git", # Example Python project
        "https://github.com/square/okhttp.git" # Example Java project
    ]
    target_languages = ["python", "java"]
    max_commits_per_repo = 50 # Limit for demonstration

    # --- Data Collection ---
    scraper = GitHubScraper(output_dir="scraped_code_samples")
    collected_bug_fix_data: List[Dict[str, Any]] = []

    for repo_url in sample_repositories:
        logger.info(f"\n--- Processing repository: {repo_url} ---")
        try:
            for i, bug_fix_pair in enumerate(
                scraper.find_bug_fix_pairs(
                    repo_url,
                    target_languages,
                    max_commits=max_commits_per_repo,
                    since_date=datetime.now() - timedelta(days=365*2) # Only commits from last 2 years
                )
            ):
                collected_bug_fix_data.append(bug_fix_pair)
                logger.info(f"  Found bug-fix pair {i+1} in {bug_fix_pair['filepath']}")
                if i >= max_commits_per_repo - 1: # Break after max_commits_per_repo pairs
                    logger.info(f"  Reached {max_commits_per_repo} pairs for {repo_url}, stopping.")
                    break
        except Exception as e:
            logger.error(f"Failed to process {repo_url}: {e}")
        finally:
            # Clean up the cloned repository after processing to save disk space
            scraper.clean_up(repo_url)

    logger.info(f"\n--- Data Collection Summary ---")
    logger.info(f"Total bug-fix pairs collected: {len(collected_bug_fix_data)}")

    # You can now save `collected_bug_fix_data` to disk (e.g., as JSON or CSV)
    # for further processing in subsequent phases.
    import json
    output_filename = os.path.join(scraper.output_dir, "bug_fix_pairs.json")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(collected_bug_fix_data, f, indent=4)
    logger.info(f"Collected data saved to: {output_filename}")


    # --- Performance Profiling Example ---
    logger.info("\n--- Running performance profiling ---")
    profile_scraper_performance(
        repo_url="https://github.com/pallets/flask.git",
        target_languages=["python"],
        max_commits=200 # Profile with more commits
    )
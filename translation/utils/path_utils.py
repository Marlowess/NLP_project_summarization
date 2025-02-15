import subprocess

def get_git_root():
    try:
        # Get the root of the git repository
        base_path = subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'], 
            universal_newlines=True
        ).strip()
        return base_path
    except subprocess.CalledProcessError:
        return None
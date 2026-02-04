"""
Archive Manager - Manages archiving and clearing of project files.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ARCHIVE_DIR = DATA_DIR / "archive"
IMAGES_DIR = DATA_DIR / "images"
AUDIO_DIR = DATA_DIR / "audio"
SCRIPTS_DIR = DATA_DIR / "scripts"
OUTPUT_DIR = DATA_DIR / "output"


def archive_current_project() -> Path:
    """
    Archive current project files to data/archive/{timestamp}/.
    
    Returns:
        Path to the archive directory
    """
    # Create archive directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_path = ARCHIVE_DIR / timestamp
    archive_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Archiving current project to: {archive_path}")
    
    # Archive images
    if IMAGES_DIR.exists() and any(IMAGES_DIR.iterdir()):
        archive_images = archive_path / "images"
        shutil.copytree(IMAGES_DIR, archive_images)
        print(f"  Archived images: {len(list(archive_images.iterdir()))} files")
    
    # Archive audio
    if AUDIO_DIR.exists() and any(AUDIO_DIR.iterdir()):
        archive_audio = archive_path / "audio"
        shutil.copytree(AUDIO_DIR, archive_audio)
        print(f"  Archived audio: {len(list(archive_audio.iterdir()))} files")
    
    # Archive scripts
    if SCRIPTS_DIR.exists() and any(SCRIPTS_DIR.iterdir()):
        archive_scripts = archive_path / "scripts"
        shutil.copytree(SCRIPTS_DIR, archive_scripts)
        print(f"  Archived scripts: {len(list(archive_scripts.iterdir()))} files")
    
    # Archive output
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        archive_output = archive_path / "output"
        shutil.copytree(OUTPUT_DIR, archive_output)
        print(f"  Archived output: {len(list(archive_output.iterdir()))} files")
    
    print(f"Archive complete: {archive_path}")
    return archive_path


def clear_working_directories():
    """
    Clear working directories for a fresh start.
    Keeps directory structure but removes files.
    """
    print("Clearing working directories...")
    
    # Clear images (keep directory)
    if IMAGES_DIR.exists():
        for file in IMAGES_DIR.iterdir():
            if file.is_file():
                file.unlink()
        print(f"  Cleared: {IMAGES_DIR}")
    
    # Clear audio (keep directory)
    if AUDIO_DIR.exists():
        for file in AUDIO_DIR.iterdir():
            if file.is_file():
                file.unlink()
        print(f"  Cleared: {AUDIO_DIR}")
    
    # Clear scripts (keep directory)
    if SCRIPTS_DIR.exists():
        for file in SCRIPTS_DIR.iterdir():
            if file.is_file():
                file.unlink()
        print(f"  Cleared: {SCRIPTS_DIR}")
    
    # Clear output (keep directory)
    if OUTPUT_DIR.exists():
        for file in OUTPUT_DIR.iterdir():
            if file.is_file():
                file.unlink()
        print(f"  Cleared: {OUTPUT_DIR}")
    
    print("Working directories cleared.")


def list_archives() -> list:
    """
    List all archived projects.
    
    Returns:
        List of archive directory paths
    """
    if not ARCHIVE_DIR.exists():
        return []
    
    archives = sorted(ARCHIVE_DIR.iterdir(), reverse=True)
    return [a for a in archives if a.is_dir()]


def has_current_project() -> bool:
    """
    Check if there's an existing project in progress.
    
    Returns:
        True if there are project files, False otherwise
    """
    # Check for script
    script_path = SCRIPTS_DIR / "current.json"
    if script_path.exists():
        return True
    
    # Check for images
    if IMAGES_DIR.exists() and any(IMAGES_DIR.glob("*.jpeg")) or any(IMAGES_DIR.glob("*.png")):
        return True
    
    # Check for audio
    audio_path = AUDIO_DIR / "current.mp3"
    if audio_path.exists():
        return True
    
    return False


def prepare_new_project(archive_existing: bool = True):
    """
    Prepare workspace for a new project.
    
    Args:
        archive_existing: If True, archive current files before clearing
    """
    if archive_existing and has_current_project():
        archive_current_project()
    
    clear_working_directories()
    
    # Ensure directories exist
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\nWorkspace ready for new project!")


def main():
    """Interactive archive manager."""
    print("=" * 50)
    print("HELAMANS VIDEOS - Archive Manager")
    print("=" * 50)
    
    if has_current_project():
        print("\n⚠️  Current project detected!")
        print("Options:")
        print("  1. Archive and start new project")
        print("  2. Continue with existing project")
        print("  3. List archived projects")
        
        choice = input("\nChoice [1/2/3]: ").strip()
        
        if choice == "1":
            prepare_new_project(archive_existing=True)
        elif choice == "2":
            print("Continuing with existing project...")
        elif choice == "3":
            archives = list_archives()
            if archives:
                print("\nArchived projects:")
                for i, archive in enumerate(archives, 1):
                    print(f"  {i}. {archive.name}")
            else:
                print("No archived projects found.")
    else:
        print("\nNo current project. Workspace is ready!")


if __name__ == "__main__":
    main()

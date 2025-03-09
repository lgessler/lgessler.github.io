#!/bin/bash

set -e  # Exit on error

# Get current branch name
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"

# Build the website for production
echo "Building the website for production..."
pelican content -o output -s publishconf.py

# Create gh-pages branch if it doesn't exist
if ! git show-ref --verify --quiet refs/heads/gh-pages; then
    echo "Creating gh-pages branch..."
    git checkout --orphan gh-pages
    git rm -rf .
    touch .nojekyll  # Prevent GitHub from using Jekyll
    git add .nojekyll
    git commit -m "Initialize gh-pages branch"
    git push origin gh-pages
    git checkout $CURRENT_BRANCH
fi

# Save the current changes to a temporary area
echo "Saving current changes..."
git stash

# Switch to gh-pages branch
echo "Switching to gh-pages branch..."
git checkout gh-pages

# Copy new content without removing existing files
echo "Copying new content to gh-pages branch..."
cp -r output/* .
touch .nojekyll  # Ensure .nojekyll file exists

# Add all changes to git
echo "Committing changes to gh-pages branch..."
git add .
git commit -m "Update website content $(date)"

# Push changes
echo "Pushing changes to gh-pages branch..."
git push origin gh-pages

# Switch back to original branch
echo "Switching back to $CURRENT_BRANCH branch..."
git checkout $CURRENT_BRANCH

# Restore stashed changes if any
git stash pop || true

echo "Deployment completed successfully!" 
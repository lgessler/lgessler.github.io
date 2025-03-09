#!/bin/bash

set -e  # Exit on error

# Get current branch name
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"

# Build the website for production
echo "Building the website for production..."
pelican content -o output -s publishconf.py

# Switch to gh-pages branch
echo "Switching to gh-pages branch..."
git checkout gh-pages

# Copy new content
echo "Copying new content..."
cp -r output/* .
rm -rf output/

# Add all changes to git
echo "Committing changes..."
git add .
git commit -m "Update website content $(date)"

# Push changes
echo "Pushing changes..."
git push origin gh-pages

# Switch back to original branch
echo "Switching back to $CURRENT_BRANCH branch..."
git checkout $CURRENT_BRANCH

echo "Deployment completed successfully!" 
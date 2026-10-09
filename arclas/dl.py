#!/usr/bin/env python3
"""
Downloads 50 color images per artist from e926.net with DEBUG output.
"""

import os
import requests
import time
from PIL import Image
from io import BytesIO
import json

# Configuration
IMAGES_PER_ARTIST = 50
OUTPUT_DIR = "images"
RESIZE_DIMENSIONS = (256, 256)
API_BASE = "https://e621.net/posts.json"

ARTISTS = [
    "chalo",
    "syuro",
    "tokifuji",
	"blacksaikou",
	"herny",
	"dracojeff",
	"frumples",
	"roly"
]

last_request_time = 0
MIN_INTERVAL = 1.0

def rate_limit():
    global last_request_time
    elapsed = time.time() - last_request_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    last_request_time = time.time()

def download_artist_images(artist: str) -> int:
    headers = {
        "User-Agent": "e926ImageDownloader/1.0 (by anonymous on e926)"
    }
    
    downloaded = 0
    page = 1
    consecutive_empty_pages = 0
    
    print(f"\n--- Downloading images for artist: {artist} ---")
    
    while downloaded < IMAGES_PER_ARTIST and consecutive_empty_pages < 3:
        params = {
            "page": page,
            "limit": 320,
            "tags": f"{artist} order:random"
        }
        
        try:
            rate_limit()
            print(f"  Fetching page {page}...", end=" ", flush=True)
            print(f"\n    URL: {API_BASE}")
            print(f"    Params: {params}")
            
            response = requests.get(API_BASE, params=params, headers=headers, timeout=15)
            
            print(f"    Status: {response.status_code}")
            
            if response.status_code == 429:
                print("RATE LIMITED! Waiting 30 seconds...")
                time.sleep(30)
                continue
            
            if response.status_code == 503:
                print("SERVICE UNAVAILABLE! Waiting 10 seconds...")
                time.sleep(10)
                continue
            
            response.raise_for_status()
            
            data = response.json()
            
            # DEBUG: Print the entire response structure
            print(f"    Response keys: {list(data.keys())}")
            print(f"    Full response (first 500 chars): {json.dumps(data, indent=2)[:500]}")
            
            # Try different ways to get posts
            posts = None
            if "posts" in data:
                posts = data["posts"]
                print(f"    Found 'posts' key: {len(posts)} items")
            elif isinstance(data, list):
                posts = data
                print(f"    Response is a list: {len(posts)} items")
            else:
                print(f"    WARNING: No 'posts' key and not a list!")
                posts = []
            
            if not posts:
                consecutive_empty_pages += 1
                print(f"    [empty]")
                page += 1
                continue
            
            consecutive_empty_pages = 0
            print(f"    [{len(posts)} posts found]")
            
            for post in posts:
                if downloaded >= IMAGES_PER_ARTIST:
                    break
                
                # Get file info (try multiple structures)
                file_info = post.get("file", {})
                file_url = file_info.get("url") if file_info else None
                
                if not file_url:
                    # Try alternate structures
                    if "preview" in post and "url" in post["preview"]:
                        file_url = post["preview"]["url"]
                    elif "image" in post and "url" in post["image"]:
                        file_url = post["image"]["url"]
                
                if not file_url:
                    continue
                
                # Skip if not an image format
                if not any(file_url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                    continue
                
                try:
                    rate_limit()
                    
                    img_response = requests.get(file_url, timeout=15, headers=headers)
                    img_response.raise_for_status()
                    
                    img = Image.open(BytesIO(img_response.content))
                    
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    
                    img = img.resize(RESIZE_DIMENSIONS, Image.LANCZOS)
                    
                    output_filename = f"{artist}_{downloaded:03d}.jpg"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    img.save(output_path, "JPEG", quality=95)
                    
                    print(f"    [{downloaded + 1:2d}/{IMAGES_PER_ARTIST}] {output_filename}")
                    
                    downloaded += 1
                    
                except Exception as e:
                    print(f"    Error processing image: {e}")
                    continue
            
            page += 1
            
        except Exception as e:
            print(f"    [ERROR: {e}]")
            time.sleep(5)
            continue
    
    print(f"✓ Downloaded {downloaded}/{IMAGES_PER_ARTIST} images for '{artist}'")
    return downloaded

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_downloaded = 0
    successful_artists = 0
    
    print(f"Starting download to: {OUTPUT_DIR}\n")
    print(f"Target: {IMAGES_PER_ARTIST} images × {len(ARTISTS)} artists")
    
    for i, artist in enumerate(ARTISTS, 1):
        print(f"\n[{i}/{len(ARTISTS)}]", end="")
        count = download_artist_images(artist)
        if count > 0:
            successful_artists += 1
            total_downloaded += count
    
    print(f"\n\n{'='*50}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*50}")
    print(f"Total images: {total_downloaded}")
    print(f"Successful artists: {successful_artists}/{len(ARTISTS)}")
    print(f"Output: {OUTPUT_DIR}/")

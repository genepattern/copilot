#!/usr/bin/env python3

import os
import time
import re
from playwright.sync_api import sync_playwright

# Configuration
GROUP_URL = "https://groups.google.com/g/genepattern-help"
OUTPUT_DIR = "library/forum/raw/"


def sanitize_filename(name):
    """Creates a safe filename from a string."""
    safe_name = re.sub(r'[\\/*?:"<>|]', "", name)
    return safe_name[:150].strip()


def scrape_thread(page, thread_url, output_dir):
    """Visits a thread URL and saves its content."""
    try:
        print(f"Scraping thread: {thread_url}")
        page.goto(thread_url, wait_until="domcontentloaded")

        # Wait for the main content to load
        try:
            page.wait_for_selector('h1', timeout=5000)
        except:
            print(f"Warning: Timeout waiting for H1 on {thread_url}")

        # Extract Title
        if page.locator('h1').count() > 0:
            title = page.locator('h1').first.inner_text()
        else:
            title = "Unknown_Title"

        # Extract Messages
        content_element = page.locator('[role="main"]')
        if not content_element.count():
            content_element = page.locator('main')

        full_text = ""
        if content_element.count():
            full_text = content_element.inner_text()
        else:
            full_text = page.locator('body').inner_text()

        # Generate Filename
        thread_id = thread_url.split('/')[-1]
        filename = f"{thread_id}_{sanitize_filename(title)}.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"URL: {thread_url}\n")
            f.write(f"Title: {title}\n")
            f.write("-" * 40 + "\n")
            f.write(full_text)

        print(f"Saved: {filename}")

    except Exception as e:
        print(f"Error scraping {thread_url}: {e}")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        print(f"Navigating to {GROUP_URL}...")
        page.goto(GROUP_URL)

        # --- Step 1: Collect all Thread URLs via Pagination ---
        thread_links = set()

        # Selector for thread links
        link_selector = 'a[href*="/g/genepattern-help/c/"]'

        # Selector for the 'Next page' button
        # We use .last to grab the one at the bottom of the list
        next_button_selector = 'div[role="button"][data-tooltip="Next page"]'

        page_num = 1
        while True:
            # 1. Grab links from current page
            page.wait_for_selector(link_selector, timeout=10000)
            elements = page.locator(link_selector).all()

            new_links_count = 0
            for el in elements:
                href = el.get_attribute("href")
                if href:
                    clean_url = href.split('?')[0]
                    if clean_url not in thread_links:
                        thread_links.add(clean_url)
                        new_links_count += 1

            print(f"Page {page_num}: Found {new_links_count} new threads (Total: {len(thread_links)})")

            # 2. Try to click 'Next'
            next_btn = page.locator(next_button_selector).last

            # Check if button exists and is not disabled (aria-disabled="true")
            if next_btn.is_visible():
                is_disabled = next_btn.get_attribute("aria-disabled")
                if is_disabled == "true":
                    print("Next button is disabled. Reached the end.")
                    break
                else:
                    print("Navigating to next page...")
                    next_btn.click()
                    page_num += 1
                    # Give time for AJAX load
                    time.sleep(3)
            else:
                print("Next button not found. Stopping.")
                break

        print(f"Collection complete. Total threads to scrape: {len(thread_links)}")

        # --- Step 2: Scrape Each Thread ---
        for i, link in enumerate(thread_links):
            full_url = link if link.startswith("http") else f"https://groups.google.com{link}"
            scrape_thread(page, full_url, OUTPUT_DIR)

            # Simple rate limiting
            time.sleep(1)

        browser.close()
        print("Done!")


if __name__ == "__main__":
    main()

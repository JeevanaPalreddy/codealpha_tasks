# Web Scraping Project: Job Listings Data Extractor
# Website: https://realpython.github.io/fake-jobs/ (Educational fake job board)
# This is a beginner-friendly web scraping project perfect for learning and LinkedIn showcase

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import csv
from datetime import datetime
import os

class JobScraper:
    def __init__(self):
        self.base_url = "https://realpython.github.io/fake-jobs/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.jobs_data = []
    
    def fetch_page(self, url):
        """Fetch webpage content with error handling"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def parse_job_listings(self, soup):
        """Extract job information from the webpage"""
        jobs = []
        
        # Find all job cards on the page
        job_cards = soup.find_all('div', class_='card-content')
        
        print(f"Found {len(job_cards)} job listings")
        
        for card in job_cards:
            try:
                # Extract job title
                title_element = card.find('h2', class_='title')
                title = title_element.get_text(strip=True) if title_element else 'N/A'
                
                # Extract company name
                company_element = card.find('h3', class_='company')
                company = company_element.get_text(strip=True) if company_element else 'N/A'
                
                # Extract location
                location_element = card.find('p', class_='location')
                location = location_element.get_text(strip=True) if location_element else 'N/A'
                
                # Extract job description
                description_element = card.find('div', class_='content')
                description = description_element.get_text(strip=True) if description_element else 'N/A'
                
                # Extract date posted (if available)
                date_element = card.find('time')
                date_posted = date_element.get('datetime') if date_element else 'N/A'
                
                # Create job dictionary
                job_data = {
                    'title': title,
                    'company': company,
                    'location': location,
                    'description': description[:200] + '...' if len(description) > 200 else description,
                    'date_posted': date_posted,
                    'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                jobs.append(job_data)
                print(f"Scraped: {title} at {company}")
                
            except Exception as e:
                print(f"Error parsing job card: {e}")
                continue
        
        return jobs
    
    def scrape_jobs(self):
        """Main scraping function"""
        print("Starting job scraping process...")
        print(f"Target website: {self.base_url}")
        
        # Fetch the main page
        response = self.fetch_page(self.base_url)
        if not response:
            print("Failed to fetch the main page")
            return []
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract job listings
        jobs = self.parse_job_listings(soup)
        
        # Store in class variable
        self.jobs_data = jobs
        
        print(f"Successfully scraped {len(jobs)} job listings")
        return jobs
    
    def save_to_csv(self, filename='scraped_jobs.csv'):
        """Save scraped data to CSV file"""
        if not self.jobs_data:
            print("No data to save. Run scrape_jobs() first.")
            return
        
        try:
            df = pd.DataFrame(self.jobs_data)
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"Data saved to {filename}")
            print(f"File location: {os.path.abspath(filename)}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")
    
    def save_to_excel(self, filename='scraped_jobs.xlsx'):
        """Save scraped data to Excel file"""
        if not self.jobs_data:
            print("No data to save. Run scrape_jobs() first.")
            return
        
        try:
            df = pd.DataFrame(self.jobs_data)
            df.to_excel(filename, index=False, engine='openpyxl')
            print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Error saving to Excel: {e}")
    
    def display_summary(self):
        """Display summary statistics of scraped data"""
        if not self.jobs_data:
            print("No data available. Run scrape_jobs() first.")
            return
        
        df = pd.DataFrame(self.jobs_data)
        
        print("\n" + "="*50)
        print("SCRAPING SUMMARY")
        print("="*50)
        print(f"Total jobs scraped: {len(df)}")
        print(f"Unique companies: {df['company'].nunique()}")
        print(f"Unique locations: {df['location'].nunique()}")
        
        print("\nTop 5 Companies by Job Count:")
        company_counts = df['company'].value_counts().head()
        for company, count in company_counts.items():
            print(f"  {company}: {count} jobs")
        
        print("\nTop 5 Locations by Job Count:")
        location_counts = df['location'].value_counts().head()
        for location, count in location_counts.items():
            print(f"  {location}: {count} jobs")
        
        print("\nSample Job Listings:")
        print(df[['title', 'company', 'location']].head().to_string(index=False))

def main():
    """Main function to run the scraping project"""
    print("Web Scraping Project: Job Listings Extractor")
    print("=" * 55)
    
    # Create scraper instance
    scraper = JobScraper()
    
    # Perform scraping
    jobs = scraper.scrape_jobs()
    
    if jobs:
        # Display summary
        scraper.display_summary()
        
        # Save data to files
        scraper.save_to_csv()
        scraper.save_to_excel()
        
        print("\n" + "="*50)
        print("PROJECT COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("Files created:")
        print("- scraped_jobs.csv (CSV format)")
        print("- scraped_jobs.xlsx (Excel format)")
        print("\nYou can now analyze this data or upload to your GitHub/LinkedIn!")
    else:
        print("No data was scraped. Please check the website and try again.")

# Additional utility functions for data analysis
def analyze_scraped_data(csv_file='scraped_jobs.csv'):
    """Analyze the scraped job data"""
    try:
        df = pd.read_csv(csv_file)
        
        print("DATA ANALYSIS REPORT")
        print("=" * 30)
        
        # Basic statistics
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for missing values
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        # Most common job titles
        print("\nMost Common Job Titles:")
        print(df['title'].value_counts().head(10))
        
        return df
    
    except FileNotFoundError:
        print(f"File {csv_file} not found. Run the scraper first.")
        return None

# Run the project
if __name__ == "__main__":
    # Install required packages first
    print("Required packages: requests, beautifulsoup4, pandas, openpyxl")
    print("Install with: pip install requests beautifulsoup4 pandas openpyxl")
    print()
    
    main()
    
    # Optional: Run analysis
    print("\nRunning data analysis...")
    analyze_scraped_data()

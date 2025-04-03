#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import os
import random
import re
import requests
import sys
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from htmldate import find_date
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, pipeline

COOKIES = {"DV": "45wk_UnjodMVQIaWaSHSPtCD9CY5Whk",
               "SOCS": "CAESNQgCEitib3FfaWRlbnRpdHlmcm9udGVuZHVpc2VydmVyXzIwMjUwMzEyLjA4X3AwGgJlbiACGgYIgMzdvgY",
               "AEC": "AVcja2f4kO11E1R_4CaXllj9dtLqClxBFyhNevtICl-fHjhaL3JyFtoxhRc",
               "__Secure-ENID": "26.SE=Ug2uXwvnMzJZ1PXcw845Oniz3cB8J3KQuwhfpI4Dq1GMuUjmFithz-PFoHF9XG1f2R17UiWYOpWaNGGzxBUbWVLOBU0r8HcYm_G2S9WAZ_mI_7f5Q37_oOVUwOnYexPz76fhAAObWA0BIsxya-HfEO1kDnbJAAX_I63vbUUm0QiRUpeTWm1RXapONN2zfMB-vpXlOwR8vNsoFDnNEUW-PukYSAuiVOb8swn7MOCqul2guF6ENGFnerWQ8wdmBDSMbe5Tg86TPeoWzXzfNCa_lV5h2Q35"}
EXCLUDE_LIST = ['maps', 'policies', 'preferences', 'accounts', 'support', 'www.google.', 'www.businesswire.com', 'www.ccn.com', 'www.makeuseof.com']
FILES = ['urls.json', 'summaries.json', 'sentiments.json']
TEMP_FILES = ['temp_urls.json', 'temp_summaries.json', 'temp_sentiments.json']
MODEL_NAME = "Nerdward/financial-summarization-pegasus-finetuned-pytorch-model"
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15'
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15'
]

logger = logging.getLogger(__name__)

def generate_date_ranges(start_date: str, end_date: str) -> list:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")   
    date_ranges = []
    current = start
    while current <= end:
        next_date = current + timedelta(days=1)
        if (end - current).days == 2:
            next_date = end
        date_ranges.append((current.strftime("%Y-%m-%d"), next_date.strftime("%Y-%m-%d")))
        current = next_date + timedelta(days=1)
    return date_ranges

def save_files(file, dictionary, save):
    with open(TEMP_FILES[file], "w") as json_file:
        json.dump(dictionary, json_file, indent=4)
    if save:
        if os.path.exists(FILES[file]) and os.stat(FILES[file]).st_size != 0:
            with open(FILES[file], "r") as json_file:
                existing_data = json.load(json_file)               
            if isinstance(existing_data, dict):
                existing_data.update(dictionary)
            else:
                raise ValueError("JSON file does not contain a dictionary.")
        else:
            existing_data = dictionary

        with open(FILES[file], "w") as json_file:
            json.dump(existing_data, json_file, indent=4)

def search_for_stock_news_links(dates):
    search_url = f'https://www.google.com/search?q=ethereum%20eth+before:{dates[1]}+after:{dates[0]}&hl=en&tbm=nws'
    session = requests.Session()
    requests.utils.add_dict_to_cookiejar(session.cookies, COOKIES)
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    r = session.get(search_url, headers=headers)
    soup = BeautifulSoup(r.text, features="html.parser")
    with open('res.html', "w") as file:
        file.write(str(soup))
    links = soup.find_all("a", {"href" : re.compile(r".*")})
    hrefs = [link['href'] for link in links]
    return hrefs

def strip_unwanted_urls(urls):
    val = []
    for url in urls:
        if 'https://' in url and not any(exc in url for exc in EXCLUDE_LIST):
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val))

def get_links(dates, save):
    logger.info('Searching for the news.')
    raw_urls = {date[0]:search_for_stock_news_links(date) for date in dates}
    logger.info('Cleaning the URLs.')
    cleaned_urls = {date[0]:strip_unwanted_urls(raw_urls[date[0]]) for date in dates}
    logger.info('Saving the URLs.')
    save_files(0, cleaned_urls, save)
    logger.info('Successfully saved the URLs.')

def scrape_and_process(session, URLs, date):
    dictionary = {}
    min_date = datetime.strptime(date, "%Y-%m-%d")
    max_date = min_date + timedelta(days=1)
    default_date = min_date.strftime("%Y-%m-%d")
    for url in URLs:
        logger.info(url)
        r = session.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, 'html.parser')
        url_date = find_date(str(soup))
        if url_date is None or not (min_date <= datetime.strptime(url_date, "%Y-%m-%d") <= max_date):
            url_date = default_date
        results = soup.find_all('p')
        text = [res.text for res in results[1:]]
        words = ' '.join(text).split(' ')[:350]
        article = ' '.join(words)
        if url_date in dictionary:
            dictionary[url_date].append((article, url))
        else:
            dictionary[url_date] = [(article, url)]
    return dictionary

def summarise(articles, tokeniser, model):
    summaries = []
    for article, url in articles:
        try:
            logger.info(url)
            input_ids = tokeniser.encode(article, return_tensors="pt")
            output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
            summary = tokeniser.decode(output[0], skip_special_tokens=True)
            if summary == "All photographs subject to copyright.":
                continue
            summaries.append((summary, url))
        except IndexError:
            continue
    return summaries

def get_summaries(session, files, save):
    logger.info('Setting up.')
    tokeniser = PegasusTokenizer.from_pretrained(MODEL_NAME)
    model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME)
    with open(files[0], "r") as json_file:
        urls = json.load(json_file)               
        if isinstance(urls, dict):
            dates = urls.keys()
        else:
            raise ValueError("JSON file does not contain a dictionary.")
    logger.info('Scraping the articles.')
    articles = {}
    for date in dates:
        articles.update(scrape_and_process(session, urls[date], date))
    logger.info('Summarising the articles.')
    summaries = {date:summarise(articles[date], tokeniser, model) for date in articles.keys()}
    logger.info('Saving the summaries.')
    save_files(1, summaries, save)
    logger.info('Successfully saved the summaries.')

def create_output_array(dates, summaries, scores):
    output = []
    for date in dates:
        for counter in range(len(summaries[date])):
            output_this = [
                            date, 
                            summaries[date][counter][0], 
                            scores[date][counter]['label'], 
                            scores[date][counter]['score'], 
                            summaries[date][counter][1]
                          ]
            output.append(output_this)
    return output

def get_sentiments(files):
    sentiment_analiser = pipeline(model='Robertuus/Crypto_Sentiment_Analysis_Bert')
    with open(files[1], "r") as json_file:
        summaries = json.load(json_file)               
        if isinstance(summaries, dict):
            dates = summaries.keys()
        else:
            raise ValueError("JSON file does not contain a dictionary.")
    scores = {date:sentiment_analiser([i[0] for i in summaries[date]]) for date in dates}              
    final_output = create_output_array(dates, summaries, scores)
    #final_output.insert(0, ['date','Summary', 'Sentiment', 'Sentiment Score', 'URL'])

    with open('ethsummaries.csv', mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(final_output)

def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--urls', help='Get links to the articles', action='store_true')
    parser.add_argument('--s_date', help='Start date', type=str)
    parser.add_argument('--e_date', help='End date', type=str)
    parser.add_argument('--summarise', help='Summarise the articles', action='store_true')
    parser.add_argument('--sentiments', help='Calculate the sentiments', action='store_true')
    parser.add_argument('--pipeline', help='Do everything', action='store_true')
    parser.add_argument('--all', help='Work on all saved data.', action='store_true')
    parser.add_argument('--save', help='Save the data.', action='store_true')
    args = parser.parse_args()

    if args.all:
        files = FILES
    else:
        files = TEMP_FILES
    session = requests.Session()
    requests.utils.add_dict_to_cookiejar(session.cookies, COOKIES)
    if args.pipeline:
        try:
            dates = generate_date_ranges(args.s_date, args.e_date)
            get_links(dates, args.save)
        except TypeError:
            logger.error('Provide the dates.')
        get_summaries(session, files, args.save)
        get_sentiments(files)
    elif args.urls:
        try:
            dates = generate_date_ranges(args.s_date, args.e_date)
            get_links(dates, args.save)
        except TypeError:
            logger.error('Provide the dates.')
    elif args.summarise:
        get_summaries(session, files, args.save)
    elif args.sentiments:
        get_sentiments(files)
    else:
        logger.info('Provide one of the options.')

if __name__ == '__main__':
    sys.exit(main())
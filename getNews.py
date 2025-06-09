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
from typing import List, Tuple, Dict, Optional, Any, Union, KeysView

from bs4 import BeautifulSoup, element
from datetime import datetime, timedelta
from htmldate import find_date # type: ignore
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, pipeline, Pipeline


COOKIES = {"DV": "45wk_UnjodMVQIaWaSHSPtCD9CY5Whk",
               "SOCS": "CAESNQgCEitib3FfaWRlbnRpdHlmcm9udGVuZHVpc2VydmVyXzIwMjUwMzEyLjA4X3AwGgJlbiACGgYIgMzdvgY",
               "AEC": "AVcja2f4kO11E1R_4CaXllj9dtLqClxBFyhNevtICl-fHjhaL3JyFtoxhRc",
               "__Secure-ENID": "26.SE=Ug2uXwvnMzJZ1PXcw845Oniz3cB8J3KQuwhfpI4Dq1GMuUjmFithz-PFoHF9XG1f2R17UiWYOpWaNGGzxBUbWVLOBU0r8HcYm_G2S9WAZ_mI_7f5Q37_oOVUwOnYexPz76fhAAObWA0BIsxya-HfEO1kDnbJAAX_I63vbUUm0QiRUpeTWm1RXapONN2zfMB-vpXlOwR8vNsoFDnNEUW-PukYSAuiVOb8swn7MOCqul2guF6ENGFnerWQ8wdmBDSMbe5Tg86TPeoWzXzfNCa_lV5h2Q35"}
EXCLUDE_LIST = ['maps', 'policies', 'preferences', 'accounts', 'support', 'www.google.', 'www.businesswire.com', 'www.ccn.com', 'www.makeuseof.com']
FILES = ['urls.json', 'summaries.json', 'sentiments.json']
TEMP_FILES = ['temp_urls.json', 'temp_summaries.json', 'temp_sentiments.json']
MODEL_NAME = "Nerdward/financial-summarization-pegasus-finetuned-pytorch-model"
SEARCH_QUERY = f'ethereum%20eth'
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

def generate_date_ranges(start_date: str, end_date: str) ->  List[Tuple[str, str]]:
    """
    Generates a list of consecutive date ranges (2-day or 3-day spans) between two dates.
    """
    start: datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end: datetime = datetime.strptime(end_date, "%Y-%m-%d")   
    date_ranges: List[Tuple[str, str]] = []
    current: datetime = start
    while current <= end:
        next_date: datetime = current + timedelta(days=1)
        if (end - current).days == 2:
            next_date = end
        date_ranges.append((current.strftime("%Y-%m-%d"), next_date.strftime("%Y-%m-%d")))
        current = next_date + timedelta(days=1)
    return date_ranges

def save_files(file: int, dictionary: Dict[str, List[Any]], save: bool) -> None:
    """
    Writes a dictionary to a temporary file and optionally merges and saves it to a main file.
    """
    with open(TEMP_FILES[file], "w") as json_file:
        json.dump(dictionary, json_file, indent=4)
    if save:
        if os.path.exists(FILES[file]) and os.stat(FILES[file]).st_size != 0:
            with open(FILES[file], "r") as json_file:
                existing_data: Dict[str, List[str]] = json.load(json_file)               
            if isinstance(existing_data, dict):
                existing_data.update(dictionary)
            else:
                raise ValueError("JSON file does not contain a dictionary.")
        else:
            existing_data: Dict[str, List[str]] = dictionary

        with open(FILES[file], "w") as json_file:
            json.dump(existing_data, json_file, indent=4)

def search_for_stock_news_links(dates: Tuple[str, str]) -> List[str]:
    """
    Searches Google News for topic-related news articles within the specified date range.
    """
    search_url: str = f'https://www.google.com/search?q={SEARCH_QUERY}+before:{dates[1]}+after:{dates[0]}&hl=en&tbm=nws'
    session: requests.Session = requests.Session()
    requests.utils.add_dict_to_cookiejar(session.cookies, COOKIES)
    headers: Dict[str, str] = {'User-Agent': random.choice(USER_AGENTS)}
    r: requests.Response = session.get(search_url, headers=headers)
    soup: BeautifulSoup = BeautifulSoup(r.text, features="html.parser")
    with open('res.html', "w") as file:
        file.write(str(soup))
    links: element.ResultSet = soup.find_all("a", {"href" : re.compile(r".*")})
    hrefs: List[str] = [link['href'] for link in links]
    return hrefs

def strip_unwanted_urls(urls: List[str]) -> List[str]:
    """
    Cleans a list of URLs by removing those that contain substrings from an exclusion list.
    """
    val: List[str] = []
    for url in urls:
        if 'https://' in url and not any(exc in url for exc in EXCLUDE_LIST):
            res: str = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val))

def get_links(dates: List[Tuple[str, str]], save: bool) -> None:
    """
    Collects, cleans, and saves news article URLs for a list of date ranges.
    """
    logger.info('Searching for the news.')
    raw_urls: Dict[str, List[str]] = {date[0]:search_for_stock_news_links(date) for date in dates}
    logger.info('Cleaning the URLs.')
    cleaned_urls: Dict[str, List[str]] = {date[0]:strip_unwanted_urls(raw_urls[date[0]]) for date in dates}
    logger.info('Saving the URLs.')
    save_files(0, cleaned_urls, save)
    logger.info('Successfully saved the URLs.')

def scrape_and_process(
    session: requests.Session,
    URLs: List[str],
    date: str
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Scrapes article content from a list of URLs.
    """
    dictionary: Dict[str, List[Tuple[str, str]]] = {}
    min_date: datetime = datetime.strptime(date, "%Y-%m-%d")
    max_date: datetime = min_date + timedelta(days=1)
    default_date: str = min_date.strftime("%Y-%m-%d")
    for url in URLs:
        logger.info(url)
        r: requests.Response = session.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup: BeautifulSoup = BeautifulSoup(r.text, 'html.parser')
        url_date: Optional[str] = find_date(str(soup))
        if url_date is None or not (min_date <= datetime.strptime(url_date, "%Y-%m-%d") <= max_date):
            url_date = default_date
        results: element.ResultSet = soup.find_all('p')
        text: List[str] = [res.text for res in results[1:]]
        words: List[str] = ' '.join(text).split(' ')[:350]
        article: str = ' '.join(words)
        if url_date in dictionary:
            dictionary[url_date].append((article, url))
        else:
            dictionary[url_date] = [(article, url)]
    return dictionary

def summarise(
    articles: List[Tuple[str, str]],
    tokeniser: PegasusTokenizer,
    model: PegasusForConditionalGeneration
) -> List[Tuple[str, str]]:
    """
    Generates short summaries for a list of article texts using a Pegasus model.
    """
    summaries: List[Tuple[str, str]] = []
    for article, url in articles:
        try:
            logger.info(url)
            input_ids = tokeniser.encode(article, return_tensors="pt")
            output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True) # type: ignore
            summary: str = tokeniser.decode(output[0], skip_special_tokens=True)
            if summary == "All photographs subject to copyright.":
                continue
            summaries.append((summary, url))
        except IndexError:
            continue
    return summaries

def get_summaries(session: requests.Session, files: List[str], save: bool) -> None:
    """
    Loads URLs from file, scrapes article content, summarizes it using Pegasus,
    and saves the results.
    """
    logger.info('Setting up.')
    tokeniser: PegasusTokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
    model: PegasusForConditionalGeneration = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME)
    with open(files[0], "r") as json_file:
        urls: Optional[Dict[str, List[str]]] = json.load(json_file)               
        if isinstance(urls, dict):
            dates: KeysView = urls.keys()
        else:
            raise ValueError("JSON file does not contain a dictionary.")
    logger.info('Scraping the articles.')
    articles: Dict[str, List[Tuple[str, str]]] = {}
    for date in dates:
        articles.update(scrape_and_process(session, urls[date], date))
    logger.info('Summarising the articles.')
    summaries: Dict[str, List[Tuple[str, str]]] = {date:summarise(articles[date], tokeniser, model) for date in articles.keys()}
    logger.info('Saving the summaries.')
    save_files(1, summaries, save)
    logger.info('Successfully saved the summaries.')

def create_output_array(
    dates: Union[List[str], Any],
    summaries: Dict[str, List[Tuple[str, str]]],
    scores: Dict[str, List[Dict[str, Union[str, float]]]]
) -> List[List[Union[str, float]]]:
    """
    Combines dates, article summaries, sentiment labels, and scores into a tabular output array.
    Returns:
        A list of lists, where each sublist contains:
        [date, summary_text, sentiment_label, sentiment_score, article_url].
    """
    output: List[List[Union[str, float]]] = []
    for date in dates:
        for counter in range(len(summaries[date])):
            output_this: List[Union[str, float]] = [
                            date, 
                            summaries[date][counter][0], 
                            scores[date][counter]['label'], 
                            scores[date][counter]['score'], 
                            summaries[date][counter][1]
                          ]
            output.append(output_this)
    return output

def get_sentiments(files: List[str]) -> None:
    """
    Loads article summaries, performs sentiment analysis, and saves the final results to a CSV file.
    """
    sentiment_analiser: Pipeline = pipeline(model='Robertuus/Crypto_Sentiment_Analysis_Bert')
    with open(files[1], "r") as json_file:
        summaries = json.load(json_file)               
        if isinstance(summaries, dict):
            dates = summaries.keys()
        else:
            raise ValueError("JSON file does not contain a dictionary.")
    scores: Dict[str, List[Dict[str, Union[str, float]]]] = {date:sentiment_analiser([i[0] for i in summaries[date]]) for date in dates} # type: ignore              
    final_output: List[List[Union[str, float]]] = create_output_array(dates, summaries, scores)
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
    session: requests.Session = requests.Session()
    requests.utils.add_dict_to_cookiejar(session.cookies, COOKIES)
    if args.pipeline:
        try:
            dates = generate_date_ranges(args.s_date, args.e_date)
            get_links(dates, args.save)
        except TypeError:
            logger.error('Provide the dates.')
            return 1
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
    return 0

if __name__ == '__main__':
    sys.exit(main())
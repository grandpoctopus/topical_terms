from typing import Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup, NavigableString

from topical_terms.build_topic_map.topic_cleanup_map import TOPIC_CLEANUP_MAP

REDDIT_URL = "https://www.reddit.com"
SUBREDDIT_LIST_URL = f"{REDDIT_URL}/r/ListOfSubreddits/wiki/listofsubreddits/"
VIDEOGAME_SUBREDDITS_LIST = f"{REDDIT_URL}/r/ListOfSubreddits/wiki/games50k"
COLUMNS = ["topic", "subreddit"]


def get_general_topics_df(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    subreddit_topics_df = pd.DataFrame(columns=COLUMNS)
    headers = soup.find_all(["h2", "h3"])
    for header in headers:
        if header.strong is not None:
            topic = header.strong.text
        elif header.em is not None:
            topic = header.em.text
        else:
            continue
        node = header.next_element.next_element.next_element.next_element
        if node is None or isinstance(node, NavigableString):
            continue
        subreddit = node.text.splitlines()
        subreddits_df = pd.DataFrame(subreddit, columns=["subreddit"])
        subreddits_df["topic"] = topic
        subreddit_topics_df = pd.concat(
            [subreddit_topics_df, subreddits_df], ignore_index=True, sort=True
        )
    return subreddit_topics_df[COLUMNS]


def get_videogame_topics_df(html: str) -> pd.DataFrame:
    vg_subreddits = []
    vg_soup = BeautifulSoup(html, "html.parser")
    wiki = vg_soup.find("div", {"class": "md wiki"})
    for p in wiki.find_all("p"):
        for a in p.find_all("a", {"rel": "nofollow"}):
            tag_text = str(a.text)
            if tag_text.startswith("/r/"):
                vg_subreddits.append(tag_text)
    vg_subs = pd.DataFrame(columns=COLUMNS)
    vg_subs["subreddit"] = vg_subreddits
    vg_subs["topic"] = "Video Games"
    return vg_subs[COLUMNS]


def get_subreddit_topics_df(general_topics_df, vg_subs_df) -> pd.DataFrame:
    subreddit_topics_df = (
        pd.concat([general_topics_df, vg_subs_df])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    subreddit_topics_df = subreddit_topics_df.groupby("subreddit")[
        "topic"
    ].apply(",".join)
    subreddit_topics_df = subreddit_topics_df.to_frame()
    subreddit_topics_df.reset_index(level=0, inplace=True)

    return subreddit_topics_df[COLUMNS]


def clean_subreddit_topics(
    subreddit_topics_df: pd.DataFrame, topic_cleanup_map: Dict
) -> pd.DataFrame:

    subreddit_substrings_to_remove = ["r/", "/"]

    for original_string in subreddit_substrings_to_remove:
        subreddit_topics_df["subreddit"] = subreddit_topics_df[
            "subreddit"
        ].str.replace(original_string, "")

    for original_string, replacement_string in topic_cleanup_map.items():
        subreddit_topics_df["topic"] = subreddit_topics_df["topic"].str.replace(
            original_string, replacement_string
        )

    return subreddit_topics_df


if __name__ == "__main__":
    subreddits_html = requests.get(SUBREDDIT_LIST_URL).text
    general_topics_df = get_general_topics_df(subreddits_html)
    vg_page = requests.get(VIDEOGAME_SUBREDDITS_LIST).text
    vg_subs_df = get_videogame_topics_df(vg_page)
    subreddit_topics_df = get_subreddit_topics_df(general_topics_df, vg_subs_df)
    clean_subreddit_topics_df = clean_subreddit_topics(
        subreddit_topics_df, TOPIC_CLEANUP_MAP
    )
    clean_subreddit_topics_df.to_csv("subreddit_topics.csv")

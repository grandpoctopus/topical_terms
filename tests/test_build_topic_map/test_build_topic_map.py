import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from topical_terms.build_topic_map.build_topic_map import (
    clean_subreddit_topics_df,
    get_general_topics_df,
    get_subreddit_topics_df,
    get_videogame_topics_df,
)


@pytest.fixture
def general_topics_html() -> str:
    ps = "<p><p><p><p><p>"
    p_closes = "</p></p></p></p></p>"
    return f"""
                <html>
                <body>
                    <h2>
                        <strong>Topic 1</strong>{ps}Subreddit 1{p_closes}</h3>
                        <h3><em>Topic 2</em>{ps}Subreddit 2{p_closes}</h3>
                        <h3><strong>Topic 2</strong>{ps}r/Subreddit 3{p_closes}</h3>
                    </h2>
                </body>
                </html>
        """


@pytest.fixture
def general_topics_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "topic": ["Topic 1", "Topic 2", "Topic 2"],
            "subreddit": ["Subreddit 1", "Subreddit 2", "r/Subreddit 3"],
        }
    )


@pytest.fixture
def video_game_topics_html() -> str:
    return """
        <html>
        <body>
            <div class="md wiki">
             <p>
            <a href="/r/gaming" rel="nofollow">/r/gaming</a></p>
            <p><a href="/r/pcgaming" rel="nofollow">/r/pcgaming</a>
            </p>
            </div>
        </body>
        </html>'
    """


@pytest.fixture
def video_game_topics_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "topic": [
                "Video Games",
                "Video Games",
            ],
            "subreddit": [
                "/r/gaming",
                "/r/pcgaming",
            ],
        }
    )


@pytest.fixture
def subreddit_topics_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "topic": [
                "Video Games",
                "Video Games",
                "Topic 1",
                "Topic 2",
                "Topic 2",
            ],
            "subreddit": [
                "/r/gaming",
                "/r/pcgaming",
                "Subreddit 1",
                "Subreddit 2",
                "r/Subreddit 3",
            ],
        }
    )


@pytest.fixture
def cleaned_topics_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "topic": [
                "Video Games",
                "Video Games",
                "Topic one",
                "Topic two",
                "Topic two",
            ],
            "subreddit": [
                "gaming",
                "pcgaming",
                "Subreddit 1",
                "Subreddit 2",
                "Subreddit 3",
            ],
        }
    )


def test_get_general_topics_df(
    general_topics_html: str, general_topics_df: pd.DataFrame
):
    actual = get_general_topics_df(general_topics_html)
    assert_frame_equal(actual, general_topics_df)


def test_get_videogame_topics_df(
    video_game_topics_html: str, video_game_topics_df
):
    actual = get_videogame_topics_df(video_game_topics_html)
    assert_frame_equal(actual, video_game_topics_df)


def test_get_subreddit_topics_df(
    general_topics_df: pd.DataFrame,
    video_game_topics_df: pd.DataFrame,
    subreddit_topics_df: pd.DataFrame,
):
    actual = get_subreddit_topics_df(general_topics_df, video_game_topics_df)
    assert_frame_equal(actual, subreddit_topics_df)


def test_clean_subreddit_topics_df(
    subreddit_topics_df: pd.DataFrame,
    cleaned_topics_df: pd.DataFrame,
):
    topic_map = {"1": "one", "2": "two"}
    actual = clean_subreddit_topics_df(subreddit_topics_df, topic_map)
    print(actual)
    assert_frame_equal(actual, cleaned_topics_df)

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark_pipeline.utilities.settings_utils import Settings
from pyspark_test import assert_pyspark_df_equal

from topical_terms.queries.topical_terms_queries import (
    TopicSpecificTrendingWordsQuery,
)
from topical_terms.schemas.topical_terms_schemas import (
    TopicSpecificTrendingWordsSchema,
)


class TestUnitTopicSpecificTrendingWordsQuery:
    @pytest.fixture()
    def query(
        self,
        local_spark: SparkSession,
        settings_obj: Settings,
        reddit_comments_df,
        subreddit_topics_map_df,
    ) -> TopicSpecificTrendingWordsQuery:

        return TopicSpecificTrendingWordsQuery(
            spark=local_spark,
            settings=settings_obj,
            schema=TopicSpecificTrendingWordsSchema(),
            subreddit_topics_map_df=subreddit_topics_map_df,
            reddit_comments_df=reddit_comments_df,
        )

    def test_run(
        self,
        query: TopicSpecificTrendingWordsQuery,
        expected_topic_specific_trending_words_df: DataFrame,
    ):

        assert_pyspark_df_equal(
            query.run(),
            expected_topic_specific_trending_words_df,
            order_by="id",
        )

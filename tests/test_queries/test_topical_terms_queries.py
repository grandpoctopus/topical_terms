import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from pyspark_pipeline.utilities.settings_utils import Settings
from pyspark_test import assert_pyspark_df_equal

from topical_terms.queries.topical_terms_queries import TopicalTermsQuery
from topical_terms.schemas.topical_terms_schemas import TopicalTermsSchema


class TestUnitTopicalTermsQuery:
    @pytest.fixture()
    def query(
        self,
        local_spark: SparkSession,
        settings_obj: Settings,
        reddit_comments_df,
        subreddit_topics_map_df,
    ) -> TopicalTermsQuery:

        return TopicalTermsQuery(
            spark=local_spark,
            settings=settings_obj,
            schema=TopicalTermsSchema(),
            subreddit_topics_map_df=subreddit_topics_map_df,
            reddit_comments_df=reddit_comments_df,
        )

    @pytest.fixture()
    def topics_column_df(self, local_spark: SparkSession) -> DataFrame:
        schema = StructType(
            [
                StructField("comment_id", StringType(), True),
                StructField("topics", StringType(), True),
                StructField("date", StringType(), True),
                StructField("body", StringType(), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "comment_id": "edgelord_1640995200",
                    "topics": "fruit,food",
                    "date": "2022-01-01",
                    "body": "banana and banana or cherry the pecan",
                },
                {
                    "comment_id": "fruitposter_1641081600",
                    "topics": "fruit,food",
                    "date": "2022-01-02",
                    "body": "cherry apple pecan",
                },
                {
                    "comment_id": "lovenuts_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "body": "walnut pecan",
                },
                {
                    "comment_id": "nutlord_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "body": "walnut pecan",
                },
                {
                    "comment_id": "nutlord_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "body": "walnut! pecan?",
                },
            ],
            schema,
        )

    @pytest.fixture()
    def tokenize_comment_body_df(self, local_spark: SparkSession) -> DataFrame:
        schema = StructType(
            [
                StructField("comment_id", StringType(), True),
                StructField("topics", StringType(), True),
                StructField("date", StringType(), True),
                StructField("words_token", ArrayType(StringType()), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "comment_id": "edgelord_1640995200",
                    "topics": "fruit,food",
                    "date": "2022-01-01",
                    "words_token": [
                        "banana",
                        "and",
                        "banana",
                        "or",
                        "cherry",
                        "the",
                        "pecan",
                    ],
                },
                {
                    "comment_id": "fruitposter_1641081600",
                    "topics": "fruit,food",
                    "date": "2022-01-02",
                    "words_token": ["cherry", "apple", "pecan"],
                },
                {
                    "comment_id": "lovenuts_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "words_token": ["walnut", "pecan"],
                },
                {
                    "comment_id": "nutlord_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "words_token": ["walnut", "pecan"],
                },
                {
                    "comment_id": "nutlord_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "words_token": ["walnut!", "pecan?"],
                },
            ],
            schema,
        )

    def test_tokenize_comment_body(
        self,
        query: TopicalTermsQuery,
        topics_column_df: DataFrame,
        tokenize_comment_body_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.tokenize_comment_body(topics_column_df),
            tokenize_comment_body_df,
            order_by="comment_id",
        )

    @pytest.fixture()
    def remove_stop_words_df(self, local_spark: SparkSession) -> DataFrame:
        schema = StructType(
            [
                StructField("comment_id", StringType(), True),
                StructField("topics", StringType(), True),
                StructField("date", StringType(), True),
                StructField("words_no_stops", ArrayType(StringType()), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "comment_id": "edgelord_1640995200",
                    "topics": "fruit,food",
                    "date": "2022-01-01",
                    "words_no_stops": ["banana", "banana", "cherry", "pecan"],
                },
                {
                    "comment_id": "fruitposter_1641081600",
                    "topics": "fruit,food",
                    "date": "2022-01-02",
                    "words_no_stops": ["cherry", "apple", "pecan"],
                },
                {
                    "comment_id": "lovenuts_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "words_no_stops": ["walnut", "pecan"],
                },
                {
                    "comment_id": "nutlord_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "words_no_stops": ["walnut", "pecan"],
                },
                {
                    "comment_id": "nutlord_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "words_no_stops": ["walnut!", "pecan?"],
                },
            ],
            schema,
        )

    def test_remove_comment_stop_words(
        self,
        query: TopicalTermsQuery,
        tokenize_comment_body_df: DataFrame,
        remove_stop_words_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.remove_comment_stop_words(tokenize_comment_body_df),
            remove_stop_words_df,
            order_by="comment_id",
        )

    @pytest.fixture()
    def word_column_df(self, local_spark: SparkSession) -> DataFrame:
        schema = StructType(
            [
                StructField("comment_id", StringType(), True),
                StructField("topics", StringType(), True),
                StructField("date", StringType(), True),
                StructField("word", StringType(), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "comment_id": "edgelord_1640995200",
                    "topics": "fruit,food",
                    "date": "2022-01-01",
                    "word": "pecan",
                },
                {
                    "comment_id": "edgelord_1640995200",
                    "topics": "fruit,food",
                    "date": "2022-01-01",
                    "word": "cherry",
                },
                {
                    "comment_id": "edgelord_1640995200",
                    "topics": "fruit,food",
                    "date": "2022-01-01",
                    "word": "banana",
                },
                {
                    "comment_id": "fruitposter_1641081600",
                    "topics": "fruit,food",
                    "date": "2022-01-02",
                    "word": "pecan",
                },
                {
                    "comment_id": "fruitposter_1641081600",
                    "topics": "fruit,food",
                    "date": "2022-01-02",
                    "word": "cherry",
                },
                {
                    "comment_id": "fruitposter_1641081600",
                    "topics": "fruit,food",
                    "date": "2022-01-02",
                    "word": "apple",
                },
                {
                    "comment_id": "lovenuts_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "word": "walnut",
                },
                {
                    "comment_id": "lovenuts_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "word": "pecan",
                },
                {
                    "comment_id": "nutlord_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "word": "pecan",
                },
                {
                    "comment_id": "nutlord_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "word": "walnut",
                },
            ],
            schema,
        )

    def test_split_words_column(
        self,
        query: TopicalTermsQuery,
        remove_stop_words_df: DataFrame,
        word_column_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.split_words_column(remove_stop_words_df),
            word_column_df,
            order_by=["comment_id", "word"],
        )

    def test_add_word_column(
        self,
        query: TopicalTermsQuery,
        topics_column_df: DataFrame,
        word_column_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.add_word_column(topics_column_df),
            word_column_df,
            order_by=["comment_id", "word"],
        )

    @pytest.fixture()
    def topic_column_df(self, local_spark: SparkSession) -> DataFrame:
        schema = StructType(
            [
                StructField("comment_id", StringType(), True),
                StructField("topic", StringType(), True),
                StructField("date", StringType(), True),
                StructField("word", StringType(), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "comment_id": "edgelord_1640995200",
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "pecan",
                },
                {
                    "comment_id": "edgelord_1640995200",
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "pecan",
                },
                {
                    "comment_id": "edgelord_1640995200",
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "cherry",
                },
                {
                    "comment_id": "edgelord_1640995200",
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "cherry",
                },
                {
                    "comment_id": "edgelord_1640995200",
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "banana",
                },
                {
                    "comment_id": "edgelord_1640995200",
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "banana",
                },
                {
                    "comment_id": "fruitposter_1641081600",
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "pecan",
                },
                {
                    "comment_id": "fruitposter_1641081600",
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "pecan",
                },
                {
                    "comment_id": "fruitposter_1641081600",
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "cherry",
                },
                {
                    "comment_id": "fruitposter_1641081600",
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "cherry",
                },
                {
                    "comment_id": "fruitposter_1641081600",
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "apple",
                },
                {
                    "comment_id": "fruitposter_1641081600",
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "apple",
                },
                {
                    "comment_id": "lovenuts_1640995200",
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "walnut",
                },
                {
                    "comment_id": "lovenuts_1640995200",
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "walnut",
                },
                {
                    "comment_id": "lovenuts_1640995200",
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "pecan",
                },
                {
                    "comment_id": "lovenuts_1640995200",
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "pecan",
                },
                {
                    "comment_id": "nutlord_1640995200",
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "pecan",
                },
                {
                    "comment_id": "nutlord_1640995200",
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "pecan",
                },
                {
                    "comment_id": "nutlord_1640995200",
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "walnut",
                },
                {
                    "comment_id": "nutlord_1640995200",
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "walnut",
                },
            ],
            schema,
        )

    def test_explode_topics_column_into_topic_column(
        self,
        query: TopicalTermsQuery,
        word_column_df: DataFrame,
        topic_column_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.explode_topics_column_into_topic_column(word_column_df),
            topic_column_df,
            order_by=["comment_id", "topic", "word"],
        )

    @pytest.fixture()
    def word_occurence_df(self, local_spark: SparkSession) -> DataFrame:
        schema = StructType(
            [
                StructField("topic", StringType(), True),
                StructField("date", StringType(), True),
                StructField("word", StringType(), True),
                StructField("daily_word_occurence", LongType(), True),
                StructField("total_daily_word_count", LongType(), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "cherry",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_occurence": 3,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_occurence": 3,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_occurence": 3,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "walnut",
                    "daily_word_occurence": 2,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "walnut",
                    "daily_word_occurence": 2,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "cherry",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_occurence": 3,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_occurence": 3,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_occurence": 3,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "walnut",
                    "daily_word_occurence": 2,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "walnut",
                    "daily_word_occurence": 2,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "apple",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 3,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "cherry",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 3,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "pecan",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 3,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "apple",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 3,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "cherry",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 3,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "pecan",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 3,
                },
            ],
            schema,
        )

    def test_add_daily_word_occurence_and_count_columns(
        self,
        query: TopicalTermsQuery,
        word_occurence_df: DataFrame,
        topic_column_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.add_daily_word_occurence_and_count_columns(topic_column_df),
            word_occurence_df,
            order_by=["date", "topic", "word"],
        )

    @pytest.fixture()
    def word_occurence_in_topic_df(
        self, local_spark: SparkSession
    ) -> DataFrame:
        schema = StructType(
            [
                StructField("topic", StringType(), True),
                StructField("date", StringType(), True),
                StructField("word", StringType(), True),
                StructField("daily_word_occurence", LongType(), True),
                StructField("total_daily_word_count", LongType(), True),
                StructField("daily_word_occurence_in_topic", LongType(), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 7,
                    "daily_word_occurence_in_topic": 1,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "cherry",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 7,
                    "daily_word_occurence_in_topic": 1,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_occurence": 3,
                    "total_daily_word_count": 7,
                    "daily_word_occurence_in_topic": 3,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "walnut",
                    "daily_word_occurence": 2,
                    "total_daily_word_count": 7,
                    "daily_word_occurence_in_topic": 2,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 7,
                    "daily_word_occurence_in_topic": 1,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "cherry",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 7,
                    "daily_word_occurence_in_topic": 1,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_occurence": 3,
                    "total_daily_word_count": 7,
                    "daily_word_occurence_in_topic": 1,
                },
                {
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_occurence": 3,
                    "total_daily_word_count": 7,
                    "daily_word_occurence_in_topic": 2,
                },
                {
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "walnut",
                    "daily_word_occurence": 2,
                    "total_daily_word_count": 7,
                    "daily_word_occurence_in_topic": 2,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "apple",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 3,
                    "daily_word_occurence_in_topic": 1,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "cherry",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 3,
                    "daily_word_occurence_in_topic": 1,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "pecan",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 3,
                    "daily_word_occurence_in_topic": 1,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "apple",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 3,
                    "daily_word_occurence_in_topic": 1,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "cherry",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 3,
                    "daily_word_occurence_in_topic": 1,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "pecan",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 3,
                    "daily_word_occurence_in_topic": 1,
                },
            ],
            schema,
        )

    def test_add_daily_word_occurence_in_topic_column(
        self,
        query: TopicalTermsQuery,
        word_occurence_df: DataFrame,
        word_occurence_in_topic_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.add_daily_word_occurence_in_topic_column(word_occurence_df),
            word_occurence_in_topic_df,
            order_by=["date", "topic", "word"],
        )

    @pytest.fixture()
    def topic_daily_word_count_df(self, local_spark: SparkSession) -> DataFrame:
        schema = StructType(
            [
                StructField("topic", StringType(), True),
                StructField("date", StringType(), True),
                StructField("word", StringType(), True),
                StructField("daily_word_occurence", LongType(), True),
                StructField("total_daily_word_count", LongType(), True),
                StructField("daily_word_occurence_in_topic", LongType(), True),
                StructField("topic_daily_word_count", LongType(), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 7,
                    "daily_word_occurence_in_topic": 1,
                    "topic_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "cherry",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 7,
                    "daily_word_occurence_in_topic": 1,
                    "topic_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_occurence": 3,
                    "total_daily_word_count": 7,
                    "daily_word_occurence_in_topic": 3,
                    "topic_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "walnut",
                    "daily_word_occurence": 2,
                    "total_daily_word_count": 7,
                    "daily_word_occurence_in_topic": 2,
                    "topic_daily_word_count": 7,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 7,
                    "daily_word_occurence_in_topic": 1,
                    "topic_daily_word_count": 3,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "cherry",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 7,
                    "daily_word_occurence_in_topic": 1,
                    "topic_daily_word_count": 3,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_occurence": 3,
                    "total_daily_word_count": 7,
                    "daily_word_occurence_in_topic": 1,
                    "topic_daily_word_count": 3,
                },
                {
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_occurence": 3,
                    "total_daily_word_count": 7,
                    "daily_word_occurence_in_topic": 2,
                    "topic_daily_word_count": 4,
                },
                {
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "walnut",
                    "daily_word_occurence": 2,
                    "total_daily_word_count": 7,
                    "daily_word_occurence_in_topic": 2,
                    "topic_daily_word_count": 4,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "apple",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 3,
                    "daily_word_occurence_in_topic": 1,
                    "topic_daily_word_count": 3,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "cherry",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 3,
                    "daily_word_occurence_in_topic": 1,
                    "topic_daily_word_count": 3,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "pecan",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 3,
                    "daily_word_occurence_in_topic": 1,
                    "topic_daily_word_count": 3,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "apple",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 3,
                    "daily_word_occurence_in_topic": 1,
                    "topic_daily_word_count": 3,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "cherry",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 3,
                    "daily_word_occurence_in_topic": 1,
                    "topic_daily_word_count": 3,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "pecan",
                    "daily_word_occurence": 1,
                    "total_daily_word_count": 3,
                    "daily_word_occurence_in_topic": 1,
                    "topic_daily_word_count": 3,
                },
            ],
            schema,
        )

    def test_add_topic_daily_word_count_column(
        self,
        query: TopicalTermsQuery,
        word_occurence_in_topic_df: DataFrame,
        topic_daily_word_count_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.add_topic_daily_word_count_column(word_occurence_in_topic_df),
            topic_daily_word_count_df,
            order_by=["date", "topic", "word"],
        )

    def test_add_word_count_columns(
        self,
        query: TopicalTermsQuery,
        topic_column_df: DataFrame,
        topic_daily_word_count_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.add_word_count_columns(topic_column_df),
            topic_daily_word_count_df,
            order_by=["date", "topic", "word"],
        )

    @pytest.fixture()
    def small_topic_daily_word_count_df(
        self, local_spark: SparkSession
    ) -> DataFrame:
        schema = StructType(
            [
                StructField("topic", StringType(), True),
                StructField("date", StringType(), True),
                StructField("word", StringType(), True),
                StructField("daily_word_occurence", LongType(), True),
                StructField("total_daily_word_count", LongType(), True),
                StructField("daily_word_occurence_in_topic", LongType(), True),
                StructField("topic_daily_word_count", LongType(), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_occurence": 2,
                    "total_daily_word_count": 20,
                    "daily_word_occurence_in_topic": 2,
                    "topic_daily_word_count": 10,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "banana",
                    "daily_word_occurence": 4,
                    "total_daily_word_count": 20,
                    "daily_word_occurence_in_topic": 3,
                    "topic_daily_word_count": 10,
                },
            ],
            schema,
        )

    @pytest.fixture()
    def frequency_and_specificity_df(
        self, local_spark: SparkSession
    ) -> DataFrame:
        schema = StructType(
            [
                StructField("topic", StringType(), True),
                StructField("date", StringType(), True),
                StructField("word", StringType(), True),
                StructField("daily_word_occurence", LongType(), True),
                StructField("total_daily_word_count", LongType(), True),
                StructField("daily_word_occurence_in_topic", LongType(), True),
                StructField("topic_daily_word_count", LongType(), True),
                StructField("frequency", DoubleType(), True),
                StructField("frequency_in_topic", DoubleType(), True),
                StructField("topic_specificity", DoubleType(), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_occurence": 2,
                    "total_daily_word_count": 20,
                    "daily_word_occurence_in_topic": 2,
                    "topic_daily_word_count": 10,
                    "frequency": 0.10,
                    "frequency_in_topic": 0.20,
                    "topic_specificity": 2.0,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "banana",
                    "daily_word_occurence": 4,
                    "total_daily_word_count": 20,
                    "daily_word_occurence_in_topic": 3,
                    "topic_daily_word_count": 10,
                    "frequency": 0.20,
                    "frequency_in_topic": 0.30,
                    "topic_specificity": 1.4999999999999998,
                },
            ],
            schema,
        )

    def test_add_topic_frequency_and_specificity_columns(
        self,
        query: TopicalTermsQuery,
        small_topic_daily_word_count_df: DataFrame,
        frequency_and_specificity_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.add_topic_frequency_and_specificity_columns(
                small_topic_daily_word_count_df
            ),
            frequency_and_specificity_df,
            order_by=["date", "topic", "word"],
        )

    @pytest.fixture()
    def rolling_average_df(self, local_spark: SparkSession) -> DataFrame:
        schema = StructType(
            [
                StructField("topic", StringType(), True),
                StructField("date", StringType(), True),
                StructField("word", StringType(), True),
                StructField("daily_word_occurence", LongType(), True),
                StructField("total_daily_word_count", LongType(), True),
                StructField("daily_word_occurence_in_topic", LongType(), True),
                StructField("topic_daily_word_count", LongType(), True),
                StructField("frequency", DoubleType(), True),
                StructField("frequency_in_topic", DoubleType(), True),
                StructField("topic_specificity", DoubleType(), True),
                StructField(
                    "five_day_average_of_frequency_in_topic", DoubleType(), True
                ),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_occurence": 2,
                    "total_daily_word_count": 20,
                    "daily_word_occurence_in_topic": 2,
                    "topic_daily_word_count": 10,
                    "frequency": 0.10,
                    "frequency_in_topic": 0.20,
                    "topic_specificity": 2.0,
                    "five_day_average_of_frequency_in_topic": 0.20,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "banana",
                    "daily_word_occurence": 4,
                    "total_daily_word_count": 20,
                    "daily_word_occurence_in_topic": 3,
                    "topic_daily_word_count": 10,
                    "frequency": 0.20,
                    "frequency_in_topic": 0.30,
                    "topic_specificity": 1.4999999999999998,
                    "five_day_average_of_frequency_in_topic": 0.25,
                },
            ],
            schema,
        )

    def test_add_add_five_day_average_of_frequency_in_topic_column(
        self,
        query: TopicalTermsQuery,
        frequency_and_specificity_df: DataFrame,
        rolling_average_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.add_five_day_average_of_frequency_in_topic_column(
                frequency_and_specificity_df
            ),
            rolling_average_df,
            order_by=["date", "topic", "word"],
        )

    @pytest.fixture()
    def change_in_rolling_average_df(
        self, local_spark: SparkSession
    ) -> DataFrame:
        schema = StructType(
            [
                StructField("topic", StringType(), True),
                StructField("date", StringType(), True),
                StructField("word", StringType(), True),
                StructField("daily_word_occurence", LongType(), True),
                StructField("total_daily_word_count", LongType(), True),
                StructField("daily_word_occurence_in_topic", LongType(), True),
                StructField("topic_daily_word_count", LongType(), True),
                StructField("frequency", DoubleType(), True),
                StructField("frequency_in_topic", DoubleType(), True),
                StructField("topic_specificity", DoubleType(), True),
                StructField(
                    "five_day_average_of_frequency_in_topic", DoubleType(), True
                ),
                StructField(
                    "change_in_average_of_frequency_in_topic",
                    DoubleType(),
                    True,
                ),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_occurence": 2,
                    "total_daily_word_count": 20,
                    "daily_word_occurence_in_topic": 2,
                    "topic_daily_word_count": 10,
                    "frequency": 0.10,
                    "frequency_in_topic": 0.20,
                    "topic_specificity": 2.0,
                    "five_day_average_of_frequency_in_topic": 0.20,
                    "change_in_average_of_frequency_in_topic": None,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "banana",
                    "daily_word_occurence": 4,
                    "total_daily_word_count": 20,
                    "daily_word_occurence_in_topic": 3,
                    "topic_daily_word_count": 10,
                    "frequency": 0.20,
                    "frequency_in_topic": 0.30,
                    "topic_specificity": 1.4999999999999998,
                    "five_day_average_of_frequency_in_topic": 0.25,
                    "change_in_average_of_frequency_in_topic": 0.04999999999999999,
                },
            ],
            schema,
        )

    def test_add_daily_change_in_average_frequency_in_topic_column(
        self,
        query: TopicalTermsQuery,
        rolling_average_df: DataFrame,
        change_in_rolling_average_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.add_daily_change_in_average_frequency_in_topic_column(
                rolling_average_df,
            ),
            change_in_rolling_average_df,
            order_by=["date", "topic", "word"],
        )

    def test_run(
        self,
        query: TopicalTermsQuery,
        expected_topic_specific_trending_words_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.run(),
            expected_topic_specific_trending_words_df,
            order_by="id",
        )

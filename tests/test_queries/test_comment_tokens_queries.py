import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, StringType, StructField, StructType
from pyspark_pipeline.utilities.settings_utils import Settings
from pyspark_test import assert_pyspark_df_equal

from topical_terms.queries.comment_tokens_queries import CommentTokensQuery
from topical_terms.schemas.topical_terms_schemas import CommentTokensSchema


class TestUnitCommentTokensQuery:
    @pytest.fixture()
    def query(
        self,
        local_spark: SparkSession,
        settings_obj: Settings,
        reddit_comments_df: DataFrame,
        subreddit_topics_map_df: DataFrame,
    ) -> CommentTokensQuery:

        return CommentTokensQuery(
            spark=local_spark,
            settings=settings_obj,
            schema=CommentTokensSchema(),
            subreddit_topics_map_df=subreddit_topics_map_df,
            reddit_comments_df=reddit_comments_df,
        )

    @pytest.fixture()
    def topics_column_df(self, local_spark: SparkSession) -> DataFrame:
        schema = StructType(
            [
                StructField("id", StringType(), True),
                StructField("topics", StringType(), True),
                StructField("date", StringType(), True),
                StructField("body", StringType(), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "id": "edgelord_1640995200",
                    "topics": "fruit,food",
                    "date": "2022-01-01",
                    "body": "banana and banana or cherry the pecan",
                },
                {
                    "id": "fruitposter_1641081600",
                    "topics": "fruit,food",
                    "date": "2022-01-02",
                    "body": "cherry apple pecan",
                },
                {
                    "id": "lovenuts_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "body": "walnut pecan",
                },
                {
                    "id": "nutlord_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "body": "walnut pecan",
                },
                {
                    "id": "nutlord_1640995200",
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
                StructField("id", StringType(), True),
                StructField("topics", StringType(), True),
                StructField("date", StringType(), True),
                StructField("words_token", ArrayType(StringType()), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "id": "edgelord_1640995200",
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
                    "id": "fruitposter_1641081600",
                    "topics": "fruit,food",
                    "date": "2022-01-02",
                    "words_token": ["cherry", "apple", "pecan"],
                },
                {
                    "id": "lovenuts_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "words_token": ["walnut", "pecan"],
                },
                {
                    "id": "nutlord_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "words_token": ["walnut", "pecan"],
                },
                {
                    "id": "nutlord_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "words_token": ["walnut!", "pecan?"],
                },
            ],
            schema,
        )

    def test_tokenize_comment_body(
        self,
        query: CommentTokensQuery,
        topics_column_df: DataFrame,
        tokenize_comment_body_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.tokenize_comment_body(topics_column_df),
            tokenize_comment_body_df,
            order_by="id",
        )

    @pytest.fixture()
    def remove_stop_words_df(self, local_spark: SparkSession) -> DataFrame:
        schema = StructType(
            [
                StructField("id", StringType(), True),
                StructField("topics", StringType(), True),
                StructField("date", StringType(), True),
                StructField("words_no_stops", ArrayType(StringType()), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "id": "edgelord_1640995200",
                    "topics": "fruit,food",
                    "date": "2022-01-01",
                    "words_no_stops": ["banana", "banana", "cherry", "pecan"],
                },
                {
                    "id": "fruitposter_1641081600",
                    "topics": "fruit,food",
                    "date": "2022-01-02",
                    "words_no_stops": ["cherry", "apple", "pecan"],
                },
                {
                    "id": "lovenuts_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "words_no_stops": ["walnut", "pecan"],
                },
                {
                    "id": "nutlord_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "words_no_stops": ["walnut", "pecan"],
                },
                {
                    "id": "nutlord_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "words_no_stops": ["walnut!", "pecan?"],
                },
            ],
            schema,
        )

    def test_remove_comment_stop_words(
        self,
        query: CommentTokensQuery,
        tokenize_comment_body_df: DataFrame,
        remove_stop_words_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.remove_comment_stop_words(tokenize_comment_body_df),
            remove_stop_words_df,
            order_by="id",
        )

    @pytest.fixture()
    def word_column_df(self, local_spark: SparkSession) -> DataFrame:
        schema = StructType(
            [
                StructField("id", StringType(), True),
                StructField("topics", StringType(), True),
                StructField("date", StringType(), True),
                StructField("word", StringType(), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "id": "edgelord_1640995200",
                    "topics": "fruit,food",
                    "date": "2022-01-01",
                    "word": "pecan",
                },
                {
                    "id": "edgelord_1640995200",
                    "topics": "fruit,food",
                    "date": "2022-01-01",
                    "word": "cherry",
                },
                {
                    "id": "edgelord_1640995200",
                    "topics": "fruit,food",
                    "date": "2022-01-01",
                    "word": "banana",
                },
                {
                    "id": "fruitposter_1641081600",
                    "topics": "fruit,food",
                    "date": "2022-01-02",
                    "word": "pecan",
                },
                {
                    "id": "fruitposter_1641081600",
                    "topics": "fruit,food",
                    "date": "2022-01-02",
                    "word": "cherry",
                },
                {
                    "id": "fruitposter_1641081600",
                    "topics": "fruit,food",
                    "date": "2022-01-02",
                    "word": "apple",
                },
                {
                    "id": "lovenuts_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "word": "walnut",
                },
                {
                    "id": "lovenuts_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "word": "pecan",
                },
                {
                    "id": "nutlord_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "word": "pecan",
                },
                {
                    "id": "nutlord_1640995200",
                    "topics": "nuts,food",
                    "date": "2022-01-01",
                    "word": "walnut",
                },
            ],
            schema,
        )

    def test_split_words_column(
        self,
        query: CommentTokensQuery,
        remove_stop_words_df: DataFrame,
        word_column_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.split_words_column(remove_stop_words_df),
            word_column_df,
            order_by=["id", "word"],
        )

    def test_add_word_column(
        self,
        query: CommentTokensQuery,
        topics_column_df: DataFrame,
        word_column_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.add_word_column(topics_column_df),
            word_column_df,
            order_by=["id", "word"],
        )

    def test_explode_topics_column_into_topic_column(
        self,
        query: CommentTokensQuery,
        word_column_df: DataFrame,
        comment_tokens_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.explode_topics_column_into_topic_column(word_column_df),
            comment_tokens_df,
            order_by=["id", "topic", "word"],
        )

    def test_run(
        self,
        query: CommentTokensQuery,
        comment_tokens_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.run(),
            comment_tokens_df,
            order_by=[
                "id",
                "topic",
                "word",
            ],
        )

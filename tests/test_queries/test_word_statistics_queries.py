import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from pyspark_pipeline.utilities.settings_utils import Settings
from pyspark_test import assert_pyspark_df_equal

from topical_terms.queries.word_statistics_queries import WordStatisticsQuery
from topical_terms.schemas.topical_terms_schemas import WordStatisticsSchema


class TestUnitWordStatisticsQuery:
    @pytest.fixture()
    def query(
        self,
        local_spark: SparkSession,
        settings_obj: Settings,
        comment_tokens_df: DataFrame,
    ) -> WordStatisticsQuery:

        return WordStatisticsQuery(
            spark=local_spark,
            settings=settings_obj,
            schema=WordStatisticsSchema(),
            comment_tokens_df=comment_tokens_df,
        )

    @pytest.fixture()
    def comment_id_tokens_df(
        self, query: WordStatisticsQuery, comment_tokens_df: DataFrame
    ) -> DataFrame:
        return query.add_comment_id_column(comment_tokens_df)

    @pytest.fixture()
    def word_frequency_df(self, local_spark: SparkSession) -> DataFrame:
        schema = StructType(
            [
                StructField("topic", StringType(), True),
                StructField("date", StringType(), True),
                StructField("word", StringType(), True),
                StructField("daily_word_frequency", LongType(), True),
                StructField("total_daily_word_count", LongType(), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "cherry",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_frequency": 3,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_frequency": 3,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_frequency": 3,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "walnut",
                    "daily_word_frequency": 2,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "walnut",
                    "daily_word_frequency": 2,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "cherry",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_frequency": 3,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_frequency": 3,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_frequency": 3,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "walnut",
                    "daily_word_frequency": 2,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "walnut",
                    "daily_word_frequency": 2,
                    "total_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "apple",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 3,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "cherry",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 3,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "pecan",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 3,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "apple",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 3,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "cherry",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 3,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "pecan",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 3,
                },
            ],
            schema,
        )

    def test_add_daily_word_frequency_and_count_columns(
        self,
        query: WordStatisticsQuery,
        word_frequency_df: DataFrame,
        comment_id_tokens_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.add_daily_word_frequency_and_count_columns(
                comment_id_tokens_df
            ),
            word_frequency_df,
            order_by=["date", "topic", "word"],
        )

    @pytest.fixture()
    def word_frequency_in_topic_df(
        self, local_spark: SparkSession
    ) -> DataFrame:
        schema = StructType(
            [
                StructField("topic", StringType(), True),
                StructField("date", StringType(), True),
                StructField("word", StringType(), True),
                StructField("daily_word_frequency", LongType(), True),
                StructField("total_daily_word_count", LongType(), True),
                StructField("daily_word_frequency_in_topic", LongType(), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 7,
                    "daily_word_frequency_in_topic": 1,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "cherry",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 7,
                    "daily_word_frequency_in_topic": 1,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_frequency": 3,
                    "total_daily_word_count": 7,
                    "daily_word_frequency_in_topic": 3,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "walnut",
                    "daily_word_frequency": 2,
                    "total_daily_word_count": 7,
                    "daily_word_frequency_in_topic": 2,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 7,
                    "daily_word_frequency_in_topic": 1,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "cherry",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 7,
                    "daily_word_frequency_in_topic": 1,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_frequency": 3,
                    "total_daily_word_count": 7,
                    "daily_word_frequency_in_topic": 1,
                },
                {
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_frequency": 3,
                    "total_daily_word_count": 7,
                    "daily_word_frequency_in_topic": 2,
                },
                {
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "walnut",
                    "daily_word_frequency": 2,
                    "total_daily_word_count": 7,
                    "daily_word_frequency_in_topic": 2,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "apple",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 3,
                    "daily_word_frequency_in_topic": 1,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "cherry",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 3,
                    "daily_word_frequency_in_topic": 1,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "pecan",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 3,
                    "daily_word_frequency_in_topic": 1,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "apple",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 3,
                    "daily_word_frequency_in_topic": 1,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "cherry",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 3,
                    "daily_word_frequency_in_topic": 1,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "pecan",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 3,
                    "daily_word_frequency_in_topic": 1,
                },
            ],
            schema,
        )

    def test_add_daily_word_frequency_in_topic_column(
        self,
        query: WordStatisticsQuery,
        word_frequency_df: DataFrame,
        word_frequency_in_topic_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.add_daily_word_frequency_in_topic_column(word_frequency_df),
            word_frequency_in_topic_df,
            order_by=["date", "topic", "word"],
        )

    @pytest.fixture()
    def topic_daily_word_count_df(self, local_spark: SparkSession) -> DataFrame:
        schema = StructType(
            [
                StructField("topic", StringType(), True),
                StructField("date", StringType(), True),
                StructField("word", StringType(), True),
                StructField("daily_word_frequency", LongType(), True),
                StructField("total_daily_word_count", LongType(), True),
                StructField("daily_word_frequency_in_topic", LongType(), True),
                StructField("topic_daily_word_count", LongType(), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 7,
                    "daily_word_frequency_in_topic": 1,
                    "topic_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "cherry",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 7,
                    "daily_word_frequency_in_topic": 1,
                    "topic_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_frequency": 3,
                    "total_daily_word_count": 7,
                    "daily_word_frequency_in_topic": 3,
                    "topic_daily_word_count": 7,
                },
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "walnut",
                    "daily_word_frequency": 2,
                    "total_daily_word_count": 7,
                    "daily_word_frequency_in_topic": 2,
                    "topic_daily_word_count": 7,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 7,
                    "daily_word_frequency_in_topic": 1,
                    "topic_daily_word_count": 3,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "cherry",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 7,
                    "daily_word_frequency_in_topic": 1,
                    "topic_daily_word_count": 3,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_frequency": 3,
                    "total_daily_word_count": 7,
                    "daily_word_frequency_in_topic": 1,
                    "topic_daily_word_count": 3,
                },
                {
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "pecan",
                    "daily_word_frequency": 3,
                    "total_daily_word_count": 7,
                    "daily_word_frequency_in_topic": 2,
                    "topic_daily_word_count": 4,
                },
                {
                    "topic": "nuts",
                    "date": "2022-01-01",
                    "word": "walnut",
                    "daily_word_frequency": 2,
                    "total_daily_word_count": 7,
                    "daily_word_frequency_in_topic": 2,
                    "topic_daily_word_count": 4,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "apple",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 3,
                    "daily_word_frequency_in_topic": 1,
                    "topic_daily_word_count": 3,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "cherry",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 3,
                    "daily_word_frequency_in_topic": 1,
                    "topic_daily_word_count": 3,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "pecan",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 3,
                    "daily_word_frequency_in_topic": 1,
                    "topic_daily_word_count": 3,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "apple",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 3,
                    "daily_word_frequency_in_topic": 1,
                    "topic_daily_word_count": 3,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "cherry",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 3,
                    "daily_word_frequency_in_topic": 1,
                    "topic_daily_word_count": 3,
                },
                {
                    "topic": "fruit",
                    "date": "2022-01-02",
                    "word": "pecan",
                    "daily_word_frequency": 1,
                    "total_daily_word_count": 3,
                    "daily_word_frequency_in_topic": 1,
                    "topic_daily_word_count": 3,
                },
            ],
            schema,
        )

    def test_add_topic_daily_word_count_column(
        self,
        query: WordStatisticsQuery,
        word_frequency_in_topic_df: DataFrame,
        topic_daily_word_count_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.add_topic_daily_word_count_column(word_frequency_in_topic_df),
            topic_daily_word_count_df,
            order_by=["date", "topic", "word"],
        )

    def test_add_word_count_columns(
        self,
        query: WordStatisticsQuery,
        comment_id_tokens_df: DataFrame,
        topic_daily_word_count_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.add_word_count_columns(comment_id_tokens_df),
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
                StructField("daily_word_frequency", LongType(), True),
                StructField("total_daily_word_count", LongType(), True),
                StructField("daily_word_frequency_in_topic", LongType(), True),
                StructField("topic_daily_word_count", LongType(), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_frequency": 2,
                    "total_daily_word_count": 20,
                    "daily_word_frequency_in_topic": 2,
                    "topic_daily_word_count": 10,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "banana",
                    "daily_word_frequency": 4,
                    "total_daily_word_count": 20,
                    "daily_word_frequency_in_topic": 3,
                    "topic_daily_word_count": 10,
                },
            ],
            schema,
        )

    @pytest.fixture()
    def rate_and_specificity_df(self, local_spark: SparkSession) -> DataFrame:
        schema = StructType(
            [
                StructField("topic", StringType(), True),
                StructField("date", StringType(), True),
                StructField("word", StringType(), True),
                StructField("daily_word_frequency", LongType(), True),
                StructField("total_daily_word_count", LongType(), True),
                StructField("daily_word_frequency_in_topic", LongType(), True),
                StructField("topic_daily_word_count", LongType(), True),
                StructField("rate", DoubleType(), True),
                StructField("rate_in_topic", DoubleType(), True),
                StructField("topic_specificity", DoubleType(), True),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_frequency": 2,
                    "total_daily_word_count": 20,
                    "daily_word_frequency_in_topic": 2,
                    "topic_daily_word_count": 10,
                    "rate": 0.10,
                    "rate_in_topic": 0.20,
                    "topic_specificity": 2.0,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "banana",
                    "daily_word_frequency": 4,
                    "total_daily_word_count": 20,
                    "daily_word_frequency_in_topic": 3,
                    "topic_daily_word_count": 10,
                    "rate": 0.20,
                    "rate_in_topic": 0.30,
                    "topic_specificity": 1.4999999999999998,
                },
            ],
            schema,
        )

    def test_add_topic_rate_and_specificity_columns(
        self,
        query: WordStatisticsQuery,
        small_topic_daily_word_count_df: DataFrame,
        rate_and_specificity_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.add_topic_rate_and_specificity_columns(
                small_topic_daily_word_count_df
            ),
            rate_and_specificity_df,
            order_by=["date", "topic", "word"],
        )

    @pytest.fixture()
    def rolling_average_df(self, local_spark: SparkSession) -> DataFrame:
        schema = StructType(
            [
                StructField("topic", StringType(), True),
                StructField("date", StringType(), True),
                StructField("word", StringType(), True),
                StructField("daily_word_frequency", LongType(), True),
                StructField("total_daily_word_count", LongType(), True),
                StructField("daily_word_frequency_in_topic", LongType(), True),
                StructField("topic_daily_word_count", LongType(), True),
                StructField("rate", DoubleType(), True),
                StructField("rate_in_topic", DoubleType(), True),
                StructField("topic_specificity", DoubleType(), True),
                StructField(
                    "five_day_average_of_rate_in_topic", DoubleType(), True
                ),
            ]
        )

        return local_spark.createDataFrame(
            [
                {
                    "topic": "food",
                    "date": "2022-01-01",
                    "word": "banana",
                    "daily_word_frequency": 2,
                    "total_daily_word_count": 20,
                    "daily_word_frequency_in_topic": 2,
                    "topic_daily_word_count": 10,
                    "rate": 0.10,
                    "rate_in_topic": 0.20,
                    "topic_specificity": 2.0,
                    "five_day_average_of_rate_in_topic": 0.20,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "banana",
                    "daily_word_frequency": 4,
                    "total_daily_word_count": 20,
                    "daily_word_frequency_in_topic": 3,
                    "topic_daily_word_count": 10,
                    "rate": 0.20,
                    "rate_in_topic": 0.30,
                    "topic_specificity": 1.4999999999999998,
                    "five_day_average_of_rate_in_topic": 0.25,
                },
            ],
            schema,
        )

    def test_add_add_five_day_average_of_rate_in_topic_column(
        self,
        query: WordStatisticsQuery,
        rate_and_specificity_df: DataFrame,
        rolling_average_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.add_five_day_average_of_rate_in_topic_column(
                rate_and_specificity_df
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
                StructField("daily_word_frequency", LongType(), True),
                StructField("total_daily_word_count", LongType(), True),
                StructField("daily_word_frequency_in_topic", LongType(), True),
                StructField("topic_daily_word_count", LongType(), True),
                StructField("rate", DoubleType(), True),
                StructField("rate_in_topic", DoubleType(), True),
                StructField("topic_specificity", DoubleType(), True),
                StructField(
                    "five_day_average_of_rate_in_topic", DoubleType(), True
                ),
                StructField(
                    "change_in_average_of_rate_in_topic",
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
                    "daily_word_frequency": 2,
                    "total_daily_word_count": 20,
                    "daily_word_frequency_in_topic": 2,
                    "topic_daily_word_count": 10,
                    "rate": 0.10,
                    "rate_in_topic": 0.20,
                    "topic_specificity": 2.0,
                    "five_day_average_of_rate_in_topic": 0.20,
                    "change_in_average_of_rate_in_topic": None,
                },
                {
                    "topic": "food",
                    "date": "2022-01-02",
                    "word": "banana",
                    "daily_word_frequency": 4,
                    "total_daily_word_count": 20,
                    "daily_word_frequency_in_topic": 3,
                    "topic_daily_word_count": 10,
                    "rate": 0.20,
                    "rate_in_topic": 0.30,
                    "topic_specificity": 1.4999999999999998,
                    "five_day_average_of_rate_in_topic": 0.25,
                    "change_in_average_of_rate_in_topic": 0.04999999999999999,
                },
            ],
            schema,
        )

    def test_add_daily_change_in_average_rate_in_topic_column(
        self,
        query: WordStatisticsQuery,
        rolling_average_df: DataFrame,
        change_in_rolling_average_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.add_daily_change_in_average_rate_in_topic_column(
                rolling_average_df,
            ),
            change_in_rolling_average_df,
            order_by=["date", "topic", "word"],
        )

    def test_run(
        self,
        query: WordStatisticsQuery,
        words_statistics_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            query.run(),
            words_statistics_df,
            order_by="id",
        )

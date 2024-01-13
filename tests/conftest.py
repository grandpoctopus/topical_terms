import os
import shutil
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from pyspark_pipeline.utilities.settings_utils import (
    Databases,
    Settings,
    SourceTable,
)


@pytest.fixture(scope="module")
def module_tmpdir():
    tmpdir = tempfile.mkdtemp()
    subdir = os.path.join(tmpdir, "sub")
    os.mkdir(subdir)
    path = os.path.join(subdir, "testCurrentTicketCount.txt")
    yield f"file://{path}"
    shutil.rmtree(tmpdir)


@pytest.fixture(scope="module")
def settings_obj(module_tmpdir) -> Settings:
    settings_obj = Settings(
        job_name="topical_terms",
        include_start_date=datetime(2022, 1, 1, tzinfo=timezone.utc),
        include_end_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        source_tables={
            "fake_table": SourceTable(
                table_type="fake_type",
                location="fake_location",
            )
        },
        spark_configs={"spark.sql.shuffle.partitions": 1},
        target_path=module_tmpdir,
        hive_output_table_type=None,
        databases=Databases(
            source_db="fake_source_db", target_db="fake_target_db"
        ),
    )

    return settings_obj


@pytest.fixture(scope="module")
def settings_obj_aws(module_tmpdir) -> Settings:
    yaml_path = Path(__file__).parent / "data" / "aws_test_settings.yaml"
    with yaml_path.open() as f:
        settings_obj = Settings(**yaml.safe_load(f))

    settings_obj.hive_output_table_type = None
    settings_obj.target_path = module_tmpdir
    return settings_obj


@pytest.fixture(scope="module")
def local_spark(settings_obj) -> SparkSession:
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    return (
        SparkSession.builder.master("local[1]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=GMT")
        .config("spark.executor.extraJavaOptions", "-Duser.timezone=GMT")
        .appName("pi_etl_pytest_local")
        .enableHiveSupport()
        .getOrCreate()
    )


@pytest.fixture(scope="module")
def handle_server():
    print(" Set Up Moto Server")
    process = subprocess.Popen(
        "moto_server s3",
        stdout=subprocess.PIPE,
        shell=True,
        preexec_fn=os.setsid,
    )

    yield
    print("Tear Down Moto Server")
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)


@pytest.fixture(scope="module")
def s3_spark(handle_server):
    # Setup spark to use s3, and point it to the moto server.

    spark_session = (
        SparkSession.builder.master("local[1]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=GMT")
        .config("spark.executor.extraJavaOptions", "-Duser.timezone=GMT")
        .appName("pi_etl_pytest_s3_local")
        .enableHiveSupport()
        .getOrCreate()
    )

    hadoop_conf = spark_session.sparkContext._jsc.hadoopConfiguration()  # NOQA
    hadoop_conf.set("fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    hadoop_conf.set("fs.s3a.access.key", "mock")
    hadoop_conf.set("fs.s3a.secret.key", "mock")
    hadoop_conf.set("fs.s3a.endpoint", "http://127.0.0.1:5000")

    time.sleep(3)
    TEST_DB_NAME = "fake_pi_etl_database"
    spark_session.sql(f"CREATE DATABASE IF NOT EXISTS {TEST_DB_NAME}")
    time.sleep(3)

    yield spark_session

    spark_session.sql(f"DROP DATABASE {TEST_DB_NAME} CASCADE")


@pytest.fixture(scope="module")
def included_date_one() -> datetime:
    return datetime(2022, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def included_date_two() -> datetime:
    return datetime(2022, 1, 2, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def excluded_date_one() -> datetime:
    return datetime(2021, 12, 31, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def excluded_date_two() -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def subreddit_topics_map_df(local_spark: SparkSession) -> DataFrame:
    schema = StructType(
        [
            StructField("subreddit", StringType(), True),
            StructField("topics", StringType(), True),
        ]
    )

    return local_spark.createDataFrame(  # type: ignore
        [
            {
                "subreddit": "fruits",
                "topics": "fruit,food",
            },
            {
                "subreddit": "nuts",
                "topics": "nuts,food",
            },
            {
                "subreddit": "aww_nuts",
                "topics": "nuts,food",
            },
            {
                "subreddit": "go_nuts",
                "topics": "nuts,food",
            },
            {
                "subreddit": "not_included",
                "topics": "nuts,food",
            },
        ],
        schema,
    )


@pytest.fixture(scope="module")
def reddit_comments_df(
    local_spark: SparkSession,
    included_date_one: datetime,
    included_date_two: datetime,
    excluded_date_one: datetime,
    excluded_date_two: datetime,
) -> DataFrame:
    def get_timestamp(dt: datetime) -> int:
        return int(time.mktime(dt.timetuple()))

    schema = StructType(
        [
            StructField("author", StringType(), True),
            StructField("body", StringType(), True),
            StructField("created_utc", StringType(), True),
            StructField("subreddit", StringType(), True),
        ]
    )

    return local_spark.createDataFrame(  # type: ignore
        [
            {
                "author": "neckbeard",
                "body": "this comment should not make it in",
                "created_utc": get_timestamp(excluded_date_one),
                "subreddit": "not_included",
            },
            {
                "author": "edgelord",
                "body": "banana and banana or cherry the pecan",
                "created_utc": get_timestamp(included_date_one),
                "subreddit": "fruits",
            },
            {
                "author": "nutlord",
                "body": "walnut pecan",
                "created_utc": get_timestamp(included_date_one),
                "subreddit": "aww_nuts",
            },
            {
                "author": "lovenuts",
                "body": "walnut pecan",
                "created_utc": get_timestamp(included_date_one),
                "subreddit": "go_nuts",
            },
            {
                "author": "fruitposter",
                "body": "cherry apple pecan",
                "created_utc": get_timestamp(included_date_two),
                "subreddit": "fruits",
            },
            {
                "author": "elderposter",
                "body": "this comment should not make it in!",
                "created_utc": get_timestamp(excluded_date_two),
                "subreddit": "not_included",
            },
        ],
        schema,
    )


@pytest.fixture(scope="module")
def words_statistics_df(
    local_spark: SparkSession,
    included_date_one: datetime,
    included_date_two: datetime,
) -> DataFrame:
    schema = StructType(
        [
            StructField(
                "change_in_average_of_rate_in_topic",
                DoubleType(),
                True,
            ),
            StructField("topic_daily_word_count", LongType(), True),
            StructField("daily_word_frequency_in_topic", LongType(), True),
            StructField("daily_word_frequency", LongType(), True),
            StructField("date", StringType(), True),
            StructField("rate", DoubleType(), True),
            StructField("rate_in_topic", DoubleType(), True),
            StructField("id", StringType(), True),
            StructField(
                "five_day_average_of_rate_in_topic", DoubleType(), True
            ),
            StructField("topic", StringType(), True),
            StructField("topic_specificity", DoubleType(), True),
            StructField("total_daily_word_count", LongType(), True),
            StructField("word", StringType(), True),
        ]
    )

    def date_string(dt: datetime) -> str:
        return str(dt.date())

    return local_spark.createDataFrame(
        [
            {
                "change_in_average_of_rate_in_topic": None,
                "topic_daily_word_count": 7,
                "daily_word_frequency_in_topic": 1,
                "daily_word_frequency": 1,
                "date": date_string(included_date_one),
                "rate": 0.14285714285714285,
                "rate_in_topic": 0.14285714285714285,
                "id": f"food_{date_string(included_date_one)}_banana",
                "five_day_average_of_rate_in_topic": 0.14285714285714285,
                "topic": "food",
                "topic_specificity": 1.0,
                "total_daily_word_count": 7,
                "word": "banana",
            },
            {
                "change_in_average_of_rate_in_topic": None,
                "topic_daily_word_count": 7,
                "daily_word_frequency_in_topic": 1,
                "daily_word_frequency": 1,
                "date": date_string(included_date_one),
                "rate": 0.14285714285714285,
                "rate_in_topic": 0.14285714285714285,
                "id": f"food_{date_string(included_date_one)}_cherry",
                "five_day_average_of_rate_in_topic": 0.14285714285714285,
                "topic": "food",
                "topic_specificity": 1.0,
                "total_daily_word_count": 7,
                "word": "cherry",
            },
            {
                "change_in_average_of_rate_in_topic": None,
                "topic_daily_word_count": 7,
                "daily_word_frequency_in_topic": 3,
                "daily_word_frequency": 3,
                "date": date_string(included_date_one),
                "rate": 0.42857142857142855,
                "rate_in_topic": 0.42857142857142855,
                "id": f"food_{date_string(included_date_one)}_pecan",
                "five_day_average_of_rate_in_topic": 0.42857142857142855,
                "topic": "food",
                "topic_specificity": 1.0,
                "total_daily_word_count": 7,
                "word": "pecan",
            },
            {
                "change_in_average_of_rate_in_topic": None,
                "topic_daily_word_count": 7,
                "daily_word_frequency_in_topic": 2,
                "daily_word_frequency": 2,
                "date": date_string(included_date_one),
                "rate": 0.2857142857142857,
                "rate_in_topic": 0.2857142857142857,
                "id": f"food_{date_string(included_date_one)}_walnut",
                "five_day_average_of_rate_in_topic": 0.2857142857142857,
                "topic": "food",
                "topic_specificity": 1.0,
                "total_daily_word_count": 7,
                "word": "walnut",
            },
            {
                "change_in_average_of_rate_in_topic": None,
                "topic_daily_word_count": 3,
                "daily_word_frequency_in_topic": 1,
                "daily_word_frequency": 1,
                "date": date_string(included_date_two),
                "rate": 0.3333333333333333,
                "rate_in_topic": 0.3333333333333333,
                "id": f"food_{date_string(included_date_two)}_apple",
                "five_day_average_of_rate_in_topic": 0.3333333333333333,
                "topic": "food",
                "topic_specificity": 1.0,
                "total_daily_word_count": 3,
                "word": "apple",
            },
            {
                "change_in_average_of_rate_in_topic": 0.0,
                "topic_daily_word_count": 3,
                "daily_word_frequency_in_topic": 1,
                "daily_word_frequency": 1,
                "date": date_string(included_date_two),
                "rate": 0.3333333333333333,
                "rate_in_topic": 0.3333333333333333,
                "id": f"fruit_{date_string(included_date_two)}_cherry",
                "five_day_average_of_rate_in_topic": 0.3333333333333333,
                "topic": "fruit",
                "topic_specificity": 1.0,
                "total_daily_word_count": 3,
                "word": "cherry",
            },
            {
                "change_in_average_of_rate_in_topic": -0.047619047619047616,
                "topic_daily_word_count": 3,
                "daily_word_frequency_in_topic": 1,
                "daily_word_frequency": 1,
                "date": date_string(included_date_two),
                "rate": 0.3333333333333333,
                "rate_in_topic": 0.3333333333333333,
                "id": f"food_{date_string(included_date_two)}_pecan",
                "five_day_average_of_rate_in_topic": 0.38095238095238093,
                "topic": "food",
                "topic_specificity": 1.0,
                "total_daily_word_count": 3,
                "word": "pecan",
            },
            {
                "change_in_average_of_rate_in_topic": None,
                "topic_daily_word_count": 3,
                "daily_word_frequency_in_topic": 1,
                "daily_word_frequency": 1,
                "date": date_string(included_date_one),
                "rate": 0.14285714285714285,
                "rate_in_topic": 0.3333333333333333,
                "id": f"fruit_{date_string(included_date_one)}_banana",
                "five_day_average_of_rate_in_topic": 0.3333333333333333,
                "topic": "fruit",
                "topic_specificity": 2.3333333333333335,
                "total_daily_word_count": 7,
                "word": "banana",
            },
            {
                "change_in_average_of_rate_in_topic": None,
                "topic_daily_word_count": 3,
                "daily_word_frequency_in_topic": 1,
                "daily_word_frequency": 1,
                "date": date_string(included_date_one),
                "rate": 0.14285714285714285,
                "rate_in_topic": 0.3333333333333333,
                "id": f"fruit_{date_string(included_date_one)}_cherry",
                "five_day_average_of_rate_in_topic": 0.3333333333333333,
                "topic": "fruit",
                "topic_specificity": 2.3333333333333335,
                "total_daily_word_count": 7,
                "word": "cherry",
            },
            {
                "change_in_average_of_rate_in_topic": None,
                "topic_daily_word_count": 3,
                "daily_word_frequency_in_topic": 1,
                "daily_word_frequency": 3,
                "date": date_string(included_date_one),
                "rate": 0.42857142857142855,
                "rate_in_topic": 0.3333333333333333,
                "id": f"fruit_{date_string(included_date_one)}_pecan",
                "five_day_average_of_rate_in_topic": 0.3333333333333333,
                "topic": "fruit",
                "topic_specificity": 0.7777777777777778,
                "total_daily_word_count": 7,
                "word": "pecan",
            },
            {
                "change_in_average_of_rate_in_topic": None,
                "topic_daily_word_count": 3,
                "daily_word_frequency_in_topic": 1,
                "daily_word_frequency": 1,
                "date": date_string(included_date_two),
                "rate": 0.3333333333333333,
                "rate_in_topic": 0.3333333333333333,
                "id": f"fruit_{date_string(included_date_two)}_apple",
                "five_day_average_of_rate_in_topic": 0.3333333333333333,
                "topic": "fruit",
                "topic_specificity": 1.0,
                "total_daily_word_count": 3,
                "word": "apple",
            },
            {
                "change_in_average_of_rate_in_topic": 0.09523809523809523,
                "topic_daily_word_count": 3,
                "daily_word_frequency_in_topic": 1,
                "daily_word_frequency": 1,
                "date": date_string(included_date_two),
                "rate": 0.3333333333333333,
                "rate_in_topic": 0.3333333333333333,
                "id": f"food_{date_string(included_date_two)}_cherry",
                "five_day_average_of_rate_in_topic": 0.23809523809523808,
                "topic": "food",
                "topic_specificity": 1.0,
                "total_daily_word_count": 3,
                "word": "cherry",
            },
            {
                "change_in_average_of_rate_in_topic": 0.0,
                "topic_daily_word_count": 3,
                "daily_word_frequency_in_topic": 1,
                "daily_word_frequency": 1,
                "date": date_string(included_date_two),
                "rate": 0.3333333333333333,
                "rate_in_topic": 0.3333333333333333,
                "id": f"fruit_{date_string(included_date_two)}_pecan",
                "five_day_average_of_rate_in_topic": 0.3333333333333333,
                "topic": "fruit",
                "topic_specificity": 1.0,
                "total_daily_word_count": 3,
                "word": "pecan",
            },
            {
                "change_in_average_of_rate_in_topic": None,
                "topic_daily_word_count": 4,
                "daily_word_frequency_in_topic": 2,
                "daily_word_frequency": 3,
                "date": date_string(included_date_one),
                "rate": 0.42857142857142855,
                "rate_in_topic": 0.5,
                "id": f"nuts_{date_string(included_date_one)}_pecan",
                "five_day_average_of_rate_in_topic": 0.5,
                "topic": "nuts",
                "topic_specificity": 1.1666666666666667,
                "total_daily_word_count": 7,
                "word": "pecan",
            },
            {
                "change_in_average_of_rate_in_topic": None,
                "topic_daily_word_count": 4,
                "daily_word_frequency_in_topic": 2,
                "daily_word_frequency": 2,
                "date": date_string(included_date_one),
                "rate": 0.2857142857142857,
                "rate_in_topic": 0.5,
                "id": f"nuts_{date_string(included_date_one)}_walnut",
                "five_day_average_of_rate_in_topic": 0.5,
                "topic": "nuts",
                "topic_specificity": 1.75,
                "total_daily_word_count": 7,
                "word": "walnut",
            },
        ],
        schema,
    )


@pytest.fixture()
def comment_tokens_df(local_spark: SparkSession) -> DataFrame:
    schema = StructType(
        [
            StructField("id", StringType(), True),
            StructField("topic", StringType(), True),
            StructField("date", StringType(), True),
            StructField("word", StringType(), True),
        ]
    )

    return local_spark.createDataFrame(
        [
            {
                "id": "edgelord_1640995200",
                "topic": "fruit",
                "date": "2022-01-01",
                "word": "pecan",
            },
            {
                "id": "edgelord_1640995200",
                "topic": "food",
                "date": "2022-01-01",
                "word": "pecan",
            },
            {
                "id": "edgelord_1640995200",
                "topic": "fruit",
                "date": "2022-01-01",
                "word": "cherry",
            },
            {
                "id": "edgelord_1640995200",
                "topic": "food",
                "date": "2022-01-01",
                "word": "cherry",
            },
            {
                "id": "edgelord_1640995200",
                "topic": "fruit",
                "date": "2022-01-01",
                "word": "banana",
            },
            {
                "id": "edgelord_1640995200",
                "topic": "food",
                "date": "2022-01-01",
                "word": "banana",
            },
            {
                "id": "fruitposter_1641081600",
                "topic": "fruit",
                "date": "2022-01-02",
                "word": "pecan",
            },
            {
                "id": "fruitposter_1641081600",
                "topic": "food",
                "date": "2022-01-02",
                "word": "pecan",
            },
            {
                "id": "fruitposter_1641081600",
                "topic": "fruit",
                "date": "2022-01-02",
                "word": "cherry",
            },
            {
                "id": "fruitposter_1641081600",
                "topic": "food",
                "date": "2022-01-02",
                "word": "cherry",
            },
            {
                "id": "fruitposter_1641081600",
                "topic": "fruit",
                "date": "2022-01-02",
                "word": "apple",
            },
            {
                "id": "fruitposter_1641081600",
                "topic": "food",
                "date": "2022-01-02",
                "word": "apple",
            },
            {
                "id": "lovenuts_1640995200",
                "topic": "nuts",
                "date": "2022-01-01",
                "word": "walnut",
            },
            {
                "id": "lovenuts_1640995200",
                "topic": "food",
                "date": "2022-01-01",
                "word": "walnut",
            },
            {
                "id": "lovenuts_1640995200",
                "topic": "food",
                "date": "2022-01-01",
                "word": "pecan",
            },
            {
                "id": "lovenuts_1640995200",
                "topic": "nuts",
                "date": "2022-01-01",
                "word": "pecan",
            },
            {
                "id": "nutlord_1640995200",
                "topic": "nuts",
                "date": "2022-01-01",
                "word": "pecan",
            },
            {
                "id": "nutlord_1640995200",
                "topic": "food",
                "date": "2022-01-01",
                "word": "pecan",
            },
            {
                "id": "nutlord_1640995200",
                "topic": "nuts",
                "date": "2022-01-01",
                "word": "walnut",
            },
            {
                "id": "nutlord_1640995200",
                "topic": "food",
                "date": "2022-01-01",
                "word": "walnut",
            },
        ],
        schema,
    )

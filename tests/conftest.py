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
                "body": "banana banana cherry pecan",
                "created_utc": get_timestamp(included_date_one),
                "subreddit": "fruits",
            },
            {
                "author": "fruitposter",
                "body": "cherry apple pecan",
                "created_utc": get_timestamp(included_date_two),
                "subreddit": "fruits",
            },
            {
                "author": "sanenut",
                "body": "walnut! pecan?",
                "created_utc": get_timestamp(included_date_one),
                "subreddit": "nuts",
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
                "author": "elderposter",
                "body": "this comment should not make it in!",
                "created_utc": get_timestamp(excluded_date_two),
                "subreddit": "not_included",
            },
        ],
        schema,
    )


@pytest.fixture(scope="module")
def expected_topic_specific_trending_words_df(
    local_spark: SparkSession,
    included_date_one: datetime,
    included_date_two: datetime,
    excluded_date_one: datetime,
    excluded_date_two: datetime,
) -> DataFrame:
    schema = StructType(
        [
            StructField(
                "change_in_rolling_average_of_daily_frequency",
                DoubleType(),
                True,
            ),
            StructField("daily_topic_word_count", LongType(), True),
            StructField("daily_word_occurence_per_topic", LongType(), True),
            StructField("daily_word_occurence", LongType(), True),
            StructField("date", StringType(), True),
            StructField("frequency", DoubleType(), True),
            StructField("frequency_in_topic", DoubleType(), True),
            StructField("id", StringType(), True),
            StructField(
                "rolling_average_of_daily_frequency", DoubleType(), True
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
                "change_in_rolling_average_of_daily_frequency": 0.2,
                "daily_topic_word_count": 10,
                "daily_word_occurence_per_topic": 2,
                "daily_word_occurence": 4,
                "date": date_string(included_date_one),
                "frequency": 0.40,
                "frequency_in_topic": 0.40,
                "id": f"food_{date_string(included_date_one)}_banana",
                "rolling_average_of_daily_frequency": 0.4,
                "topic": "food",
                "topic_specificity": 1.0,
                "total_daily_word_count": 48,
                "word": "banana",
            },
            {
                "change_in_rolling_average_of_daily_frequency": 0.0,
                "daily_topic_word_count": 10,
                "daily_word_occurence_per_topic": 1,
                "daily_word_occurence": 2,
                "date": date_string(included_date_one),
                "frequency": 0.40,
                "frequency_in_topic": 0.40,
                "id": f"food_{date_string(included_date_one)}_cherry",
                "rolling_average_of_daily_frequency": 0.4,
                "topic": "food",
                "topic_specificity": 1.0,
                "total_daily_word_count": 48,
                "word": "cherry",
            },
            {
                "change_in_rolling_average_of_daily_frequency": 0.0,
                "daily_topic_word_count": 10,
                "daily_word_occurence_per_topic": 4,
                "daily_word_occurence": 8,
                "date": date_string(included_date_one),
                "frequency": 0.40,
                "frequency_in_topic": 0.40,
                "id": f"food_{date_string(included_date_one)}_pecan",
                "rolling_average_of_daily_frequency": 0.4,
                "topic": "food",
                "topic_specificity": 1.0,
                "total_daily_word_count": 48,
                "word": "pecan",
            },
            {
                "change_in_rolling_average_of_daily_frequency": 0.0,
                "daily_topic_word_count": 10,
                "daily_word_occurence_per_topic": 3,
                "daily_word_occurence": 6,
                "date": date_string(included_date_one),
                "frequency": 0.40,
                "frequency_in_topic": 0.40,
                "id": f"food_{date_string(included_date_one)}_walnut",
                "rolling_average_of_daily_frequency": 0.4,
                "topic": "food",
                "topic_specificity": 1.0,
                "total_daily_word_count": 48,
                "word": "walnut",
            },
            {
                "change_in_rolling_average_of_daily_frequency": 0.0,
                "daily_topic_word_count": 10,
                "daily_word_occurence_per_topic": 1,
                "daily_word_occurence": 2,
                "date": date_string(included_date_one),
                "frequency": 0.40,
                "frequency_in_topic": 0.40,
                "id": f"food_{date_string(included_date_one)}_walnut",
                "rolling_average_of_daily_frequency": 0.4,
                "topic": "food",
                "topic_specificity": 1.0,
                "total_daily_word_count": 12,
                "word": "apple",
            },
        ],
        schema,
    )

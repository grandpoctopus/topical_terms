from argparse import Namespace
from datetime import date

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark_pipeline.utilities.settings_utils import Settings, SourceTable
from pyspark_test import assert_pyspark_df_equal

from topical_terms.jobs.topical_terms_job import TopicalTermsJob
from topical_terms.schemas.topical_terms_schemas import TopicalTermsSchema


class TestUnitTopicalTermsQuery:
    @pytest.fixture()
    def job(
        self,
        local_spark: SparkSession,
        settings_obj: Settings,
        reddit_comments_df,
        subreddit_topics_map_df,
        tmpdir,
    ) -> TopicalTermsJob:

        args = Namespace()
        args.incremental_processing_type = None
        args.incremental_load_start_date = str(date(1900, 1, 1))
        args.incremental_load_end_date = str(date(2100, 1, 1))
        settings_obj.source_tables["audit_df"] = SourceTable(
            table_type="parquet", location=f"file:///{str(tmpdir)}"
        )

        return TopicalTermsJob(
            spark=local_spark,
            settings=settings_obj,
            audit_df=None,
            args=args,
            schema=TopicalTermsSchema(),
            subreddit_topics_map_df=subreddit_topics_map_df,
            reddit_comments_df=reddit_comments_df,
        )

    def test_run(
        self,
        job: TopicalTermsJob,
        expected_topic_specific_trending_words_df: DataFrame,
    ):
        assert_pyspark_df_equal(
            job.run()["topical_terms"],
            expected_topic_specific_trending_words_df,
            order_by="id",
        )

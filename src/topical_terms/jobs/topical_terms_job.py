from typing import Dict

from pyspark.sql import DataFrame
from pyspark.sql.utils import AnalysisException
from pyspark_pipeline.jobs import Job
from pyspark_pipeline.utilities.settings_utils import Settings, get_table_name

from topical_terms.queries.topical_terms_queries import TopicalTermsQuery
from topical_terms.schemas.topical_terms_schemas import TopicalTermsSchema


class TopicalTermsJob(Job):
    def __init__(
        self,
        reddit_comments_df: DataFrame,
        subreddit_topics_map_df: DataFrame,
        audit_df: DataFrame,
        **kwargs,
    ):
        """
        args:
            df: A spark DataFrame containing reddit comments
                along with subreddit information and timestamps
            subreddit_topics_map_df: a mapping between subreddits and topics
                discussed in that subreddit
        """
        super().__init__(**kwargs)
        self.reddit_comments_df = reddit_comments_df
        self.subreddit_topics_map_df = subreddit_topics_map_df
        if audit_df is None:
            try:
                audit_df = self.spark.table(
                    get_table_name("topical_terms_audit", self.settings)
                )
            except AnalysisException:
                pass

        self.audit_df = audit_df

    def run(self) -> Dict[str, DataFrame]:
        settings: Settings = self.settings
        if settings.incremental_processing_type is None:
            self.audit_df = None
        if self.logger is not None:
            self.logger.info("Running job with settings: %s " % settings)

        incremental_processing_type = settings.incremental_processing_type

        topical_terms_df = TopicalTermsQuery(
            spark=self.spark,
            settings=settings,
            schema=TopicalTermsSchema(),
            subreddit_topics_map_df=self.subreddit_topics_map_df,
            reddit_comments_df=self.reddit_comments_df,
        ).write(
            "topical_terms",
            "parquet",
            self.logger,
            row_id_columns=["id"],
            hive_table_type=settings.hive_output_table_type,
            col_to_repartition_by="id",
            partition_hdfs_by_col="id",
            incremental_processing_type=incremental_processing_type,
        )

        self.write_audit("sdoh_audit", settings)

        return {"topical_terms": topical_terms_df}

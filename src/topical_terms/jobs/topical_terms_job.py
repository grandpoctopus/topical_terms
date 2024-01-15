from typing import Dict

from pyspark.sql import DataFrame
from pyspark.sql.utils import AnalysisException
from pyspark_pipeline.jobs import Job
from pyspark_pipeline.utilities.settings_utils import Settings, get_table_name

from topical_terms.queries.comment_tokens_queries import CommentTokensQuery
from topical_terms.queries.word_statistics_queries import WordStatisticsQuery
from topical_terms.schemas.topical_terms_schemas import (
    CommentTokensSchema,
    WordStatisticsSchema,
)


class TopicalTermsJob(Job):
    def __init__(
        self,
        reddit_comments_df: DataFrame,
        subreddit_topics_map_df: DataFrame,
        audit_df: DataFrame,
        settings: Settings,
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
        self.settings = settings
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

        comment_tokens_df = CommentTokensQuery(
            spark=self.spark,
            settings=settings,
            schema=CommentTokensSchema(),
            subreddit_topics_map_df=self.subreddit_topics_map_df,
            reddit_comments_df=self.reddit_comments_df,
        ).run()

        comment_tokens_df.cache()

        topical_terms_df = WordStatisticsQuery(
            spark=self.spark,
            settings=settings,
            schema=WordStatisticsSchema(),
            comment_tokens_df=comment_tokens_df,
        ).write(
            "topical_terms",
            "parquet",
            self.logger,
            row_id_columns=["id"],
            hive_table_type=settings.hive_output_table_type,
            col_to_repartition_by=["topic", "date"],
            partition_hdfs_by_col=["topic", "date"],
            incremental_processing_type=incremental_processing_type,
        )

        comment_tokens_df.unpersist()

        self.write_audit("sdoh_audit", settings)

        return {
            "comment_tokens": comment_tokens_df,
            "topical_terms": topical_terms_df,
        }

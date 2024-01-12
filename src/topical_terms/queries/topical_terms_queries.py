from typing import List

from pyspark.ml.feature import StopWordsRemover, Tokenizer
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    avg,
    broadcast,
    col,
    concat,
    count,
    explode,
    from_unixtime,
    lag,
    lit,
    lower,
    split,
    sum,
    to_date,
    trim,
)
from pyspark.sql.window import Window
from pyspark_pipeline.queries import Query


class TopicSpecificTrendingWordsQuery(Query):
    def __init__(
        self,
        reddit_comments_df: DataFrame,
        subreddit_topics_map_df: DataFrame,
        **kwargs,
    ):
        """
        args:
            df: A spark DataFrame containing reddit comments
                along with subreddit information and timestamps
            subreddit_topics_map_df: a mapping between subreddits and topics
                discussed in that subreddit
        """

        self.reddit_comments_df = reddit_comments_df
        self.subreddit_topics_map_df = subreddit_topics_map_df
        super().__init__(**kwargs)

    def add_comment_id_column(self, df: DataFrame) -> DataFrame:
        return df.withColumn(
            "comment_id", concat(col("author"), lit("_"), col("created_utc"))
        ).select(
            "body",
            "comment_id",
            "created_utc",
            "subreddit",
        )

    def add_date_columns(self, df: DataFrame) -> DataFrame:
        """
        Comments are aggregated with different date grain
        this method creates all of the date columns at these
        different grains.
        """
        return (
            df.withColumn("date_time", from_unixtime("created_utc"))
            .withColumn("date", to_date(col("date_time")))
            .select(
                "comment_id",
                "body",
                "date",
                "subreddit",
            )
        )

    def filter_by_eligibility_dates(self, df: DataFrame) -> DataFrame:
        return df.where(
            col("date").between(
                self.settings.include_start_date,
                self.settings.include_end_date,
            )
        )

    def add_topics_column(
        self,
        df: DataFrame,
    ) -> DataFrame:
        """
        Adds the topic column from subreddits_topics_map_df
        and splits topics to individual rows. This function
        will increase the number of rows.
        """
        df = df.withColumn("subreddit", lower(col("subreddit")))
        subreddit_topic_map_df = self.subreddit_topics_map_df.withColumn(
            "subreddit", lower(col("subreddit"))
        )

        return df.join(
            broadcast(subreddit_topic_map_df),
            on=["subreddit"],
            how="inner",
        ).select(
            "comment_id",
            "body",
            "date",
            "subreddit",
            "topics",
        )

    def tokenize_comment_body(self, df: DataFrame) -> DataFrame:
        """
        splits the comments into individual words
        """
        df = df.withColumn("body", lower(col("body")))
        tokenizer = Tokenizer(inputCol="body", outputCol="words_token")
        return tokenizer.transform(df).select(
            "comment_id", "topics", "date", "words_token"
        )

    def remove_comment_stop_words(self, df: DataFrame) -> DataFrame:
        """
        Removes stop words like 'and', 'the' etc. from comments
        """
        remover = StopWordsRemover(
            inputCol="words_token", outputCol="words_no_stops"
        )

        return remover.transform(df).select(
            "comment_id",
            "topics",
            "date",
            "words_no_stops",
        )

    def split_words_column(self, df: DataFrame) -> DataFrame:
        """
        Some tokens still contain punctuation this method
        further cleans punctuation and produces a cleaned 'word' column

        Drop duplicates
        """
        return (
            df.withColumn("words_and_punct", explode("words_no_stops"))
            .withColumn(
                "raw_word", explode(split(col("words_and_punct"), "[\\W_]+"))
            )
            .withColumn("word", trim(col("raw_word")))
            .where(col("word").rlike("[a-zA-Z]"))
            .select(
                "comment_id",
                "topics",
                "word",
                "date",
            )
            .distinct()
        )

    def add_word_column(self, df: DataFrame) -> DataFrame:
        return (
            df.transform(self.tokenize_comment_body)
            .transform(self.remove_comment_stop_words)
            .transform(self.split_words_column)
            .select("comment_id", "topics", "word", "date")
        )

    def explode_topics_column_into_topic_column(
        self, df: DataFrame
    ) -> DataFrame:
        """
        Some subreddits relate to multiple topics separated by commas
        """

        return (
            df.withColumn("raw_topic", explode(split(col("topics"), ",")))
            .withColumn("topic", trim(lower(col("raw_topic"))))
            .select(
                "comment_id",
                "topic",
                "word",
                "date",
            )
        )

    def sum_column(
        self,
        df: DataFrame,
        col_to_sum: str,
        group_by_cols: List[str],
        new_col_name: str,
    ) -> DataFrame:
        partition_num = self.settings.spark_configs.get(
            "spark.sql.shuffle.partitions", 9600
        )
        df = df.repartition(partition_num, group_by_cols)

        sum_df = df.groupby(*group_by_cols).agg(
            sum(col_to_sum).alias(new_col_name)
        )

        return df.join(
            sum_df,
            on=group_by_cols,
            how="inner",
        )

    def get_daily_word_occurence_and_count(self, df: DataFrame) -> DataFrame:
        partition_num = self.settings.spark_configs.get(
            "spark.sql.shuffle.partitions", 9600
        )
        df = df.repartition(partition_num, ["date", "word"])

        word_occurence_df = (
            df.select(
                "comment_id",
                "word",
                "date",
            )
            .distinct()
            .groupBy(
                "date",
                "word",
            )
            .agg(count("word").alias("daily_word_occurence"))
            .select("date", "word", "daily_word_occurence")
        )

        total_daily_word_count_df = self.sum_column(
            word_occurence_df,
            col_to_sum="daily_word_occurence",
            group_by_cols=["date"],
            new_col_name="total_daily_word_count",
        ).select(
            "word",
            "date",
            "daily_word_occurence",
            "total_daily_word_count",
        )

        return df.join(
            total_daily_word_count_df, on=["date", "word"], how="inner"
        ).select(
            "date",
            "word",
            "topic",
            "daily_word_occurence",
            "total_daily_word_count",
        )

    def get_daily_word_occurence_per_topic(self, df: DataFrame) -> DataFrame:
        return (
            df.groupBy(
                "date",
                "word",
                "daily_word_occurence",
                "total_daily_word_count",
                "topic",
            )
            .agg(count("word").alias("daily_word_occurence_per_topic"))
            .select(
                "topic",
                "word",
                "date",
                "daily_word_occurence",
                "daily_word_occurence_per_topic",
                "total_daily_word_count",
            )
        )

    def get_topic_daily_word_count(self, df: DataFrame) -> DataFrame:
        return self.sum_column(
            df,
            col_to_sum="daily_word_occurence_per_topic",
            group_by_cols=["date", "topic"],
            new_col_name="daily_topic_word_count",
        ).select(
            "topic",
            "word",
            "date",
            "daily_word_occurence_per_topic",
            "daily_word_occurence",
            "total_daily_word_count",
            "daily_topic_word_count",
        )

    def add_word_count_columns(self, df: DataFrame) -> DataFrame:
        """
        Add columns to store a series of counts necessary
        for computing changes in word frequencies

        NOTE: the order of transformations minimizes the
        number of shuffles needed to complete the counts
        """
        return (
            df.transform(self.get_daily_word_occurence_and_count)
            .transform(self.get_daily_word_occurence_per_topic)
            .transform(self.get_topic_daily_word_count)
            .select(
                "topic",
                "word",
                "date",
                "daily_word_occurence_per_topic",
                "daily_word_occurence",
                "total_daily_word_count",
                "daily_topic_word_count",
            )
        )

    def add_topic_frequency_and_specificity_columns(
        self, df: DataFrame
    ) -> DataFrame:
        return (
            df.withColumn(
                "frequency",
                ((col("daily_word_occurence") / col("total_daily_word_count"))),
            )
            .withColumn(
                "frequency_in_topic",
                (
                    (
                        col("daily_word_occurence_per_topic")
                        / col("daily_topic_word_count")
                    )
                ),
            )
            .withColumn(
                "topic_specificity",
                (col("frequency_in_topic")) / (col("frequency")),
            )
            .select(
                "topic",
                "word",
                "date",
                "daily_word_occurence_per_topic",
                "daily_word_occurence",
                "total_daily_word_count",
                "daily_topic_word_count",
                "frequency",
                "frequency_in_topic",
                "topic_specificity",
            )
        )

    def add_rolling_average_daily_frequency_column(
        self, df: DataFrame
    ) -> DataFrame:

        four_day_window = (
            Window.partitionBy(["topic", "word"])
            .orderBy(col("date"))
            .rowsBetween(-4, 0)
        )

        return df.withColumn(
            "rolling_average_of_daily_frequency",
            avg("frequency_in_topic").over(four_day_window),
        ).select(
            "topic",
            "word",
            "date",
            "daily_word_occurence_per_topic",
            "daily_word_occurence",
            "total_daily_word_count",
            "daily_topic_word_count",
            "frequency",
            "frequency_in_topic",
            "topic_specificity",
            "rolling_average_of_daily_frequency",
        )

    def add_change_in_rolling_average_column(self, df: DataFrame) -> DataFrame:
        one_day_window = Window.partitionBy(["topic", "word"]).orderBy("date")

        return (
            df.withColumn(
                "prev_day_rolling_average",
                lag(df["rolling_average_of_daily_frequency"]).over(
                    one_day_window
                ),
            )
            .withColumn(
                "change_in_rolling_average_of_daily_frequency",
                (
                    (
                        col("rolling_average_of_daily_frequency")
                        - col("prev_day_rolling_average")
                    )
                ),
            )
            .select(
                "topic",
                "word",
                "date",
                "daily_word_occurence_per_topic",
                "daily_word_occurence",
                "total_daily_word_count",
                "daily_topic_word_count",
                "frequency",
                "frequency_in_topic",
                "topic_specificity",
                "rolling_average_of_daily_frequency",
                "change_in_rolling_average_of_daily_frequency",
            )
        )

    def add_id_column(self, df: DataFrame) -> DataFrame:
        return df.withColumn(
            "id",
            concat(col("topic"), lit("_"), col("date"), lit("_"), col("word")),
        ).select(
            "topic",
            "word",
            "date",
            "daily_word_occurence_per_topic",
            "daily_word_occurence",
            "total_daily_word_count",
            "daily_topic_word_count",
            "frequency",
            "frequency_in_topic",
            "topic_specificity",
            "rolling_average_of_daily_frequency",
            "change_in_rolling_average_of_daily_frequency",
            "id",
        )

    def run(self):

        word_usage_stats_df = (
            self.reddit_comments_df.select(
                "author", "body", "created_utc", "subreddit"
            )
            .transform(self.add_comment_id_column)
            .transform(self.add_date_columns)
            .transform(self.filter_by_eligibility_dates)
            .transform(self.add_topics_column)
            .transform(self.add_word_column)
            .transform(self.explode_topics_column_into_topic_column)
            .transform(self.add_word_count_columns)
            .transform(self.add_topic_frequency_and_specificity_columns)
            .transform(self.add_rolling_average_daily_frequency_column)
            .transform(self.add_change_in_rolling_average_column)
            .transform(self.add_id_column)
            .select(
                "id",
                "topic",
                "word",
                "date",
                "daily_word_occurence_per_topic",
                "daily_word_occurence",
                "total_daily_word_count",
                "daily_topic_word_count",
                "frequency",
                "frequency_in_topic",
                "topic_specificity",
                "rolling_average_of_daily_frequency",
                "change_in_rolling_average_of_daily_frequency",
            )
        )

        return self.enforce_schema_and_uniqueness(
            word_usage_stats_df, self.schema
        )

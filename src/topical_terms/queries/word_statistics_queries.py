from typing import List

from pyspark.sql import DataFrame
from pyspark.sql.functions import avg, col, concat, count, lag, lit, sum
from pyspark.sql.window import Window
from pyspark_pipeline.queries import Query


class WordStatisticsQuery(Query):
    def __init__(
        self,
        comment_tokens_df: DataFrame,
        **kwargs,
    ):
        """
        args:
            comment_tokens_df: a dataframe of tokenized comments
            with: 'word', 'topic', 'date', 'id' columns
        """

        self.comment_tokens_df = comment_tokens_df
        super().__init__(**kwargs)

    def add_comment_id_column(self, df: DataFrame) -> DataFrame:
        return df.withColumnRenamed("id", "comment_id")

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

    def add_daily_word_occurence_and_count_columns(
        self, df: DataFrame
    ) -> DataFrame:
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

    def add_daily_word_occurence_in_topic_column(
        self, df: DataFrame
    ) -> DataFrame:

        return (
            df.groupBy(
                "date",
                "word",
                "daily_word_occurence",
                "total_daily_word_count",
                "topic",
            )
            .agg(count("word").alias("daily_word_occurence_in_topic"))
            .select(
                "topic",
                "word",
                "date",
                "daily_word_occurence",
                "daily_word_occurence_in_topic",
                "total_daily_word_count",
            )
        )

    def add_topic_daily_word_count_column(self, df: DataFrame) -> DataFrame:
        return self.sum_column(
            df,
            col_to_sum="daily_word_occurence_in_topic",
            group_by_cols=["date", "topic"],
            new_col_name="topic_daily_word_count",
        ).select(
            "topic",
            "word",
            "date",
            "daily_word_occurence_in_topic",
            "daily_word_occurence",
            "total_daily_word_count",
            "topic_daily_word_count",
        )

    def add_word_count_columns(self, df: DataFrame) -> DataFrame:
        """
        Add columns to store a series of counts necessary
        for computing changes in word frequencies

        NOTE: the order of transformations minimizes the
        number of shuffles needed to complete the counts
        """
        return (
            df.transform(self.add_daily_word_occurence_and_count_columns)
            .transform(self.add_daily_word_occurence_in_topic_column)
            .transform(self.add_topic_daily_word_count_column)
            .select(
                "topic",
                "word",
                "date",
                "daily_word_occurence_in_topic",
                "daily_word_occurence",
                "total_daily_word_count",
                "topic_daily_word_count",
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
                        col("daily_word_occurence_in_topic")
                        / col("topic_daily_word_count")
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
                "daily_word_occurence_in_topic",
                "daily_word_occurence",
                "total_daily_word_count",
                "topic_daily_word_count",
                "frequency",
                "frequency_in_topic",
                "topic_specificity",
            )
        )

    def add_five_day_average_of_frequency_in_topic_column(
        self, df: DataFrame
    ) -> DataFrame:

        four_day_window = (
            Window.partitionBy(["topic", "word"])
            .orderBy(col("date"))
            .rowsBetween(-4, 0)
        )

        return df.withColumn(
            "five_day_average_of_frequency_in_topic",
            avg("frequency_in_topic").over(four_day_window),
        ).select(
            "topic",
            "word",
            "date",
            "daily_word_occurence_in_topic",
            "daily_word_occurence",
            "total_daily_word_count",
            "topic_daily_word_count",
            "frequency",
            "frequency_in_topic",
            "topic_specificity",
            "five_day_average_of_frequency_in_topic",
        )

    def add_daily_change_in_average_frequency_in_topic_column(
        self, df: DataFrame
    ) -> DataFrame:
        one_day_window = Window.partitionBy(["topic", "word"]).orderBy("date")

        return (
            df.withColumn(
                "prev_day_rolling_average",
                lag(df["five_day_average_of_frequency_in_topic"]).over(
                    one_day_window
                ),
            )
            .withColumn(
                "change_in_average_of_frequency_in_topic",
                (
                    (
                        col("five_day_average_of_frequency_in_topic")
                        - col("prev_day_rolling_average")
                    )
                ),
            )
            .select(
                "topic",
                "word",
                "date",
                "daily_word_occurence_in_topic",
                "daily_word_occurence",
                "total_daily_word_count",
                "topic_daily_word_count",
                "frequency",
                "frequency_in_topic",
                "topic_specificity",
                "five_day_average_of_frequency_in_topic",
                "change_in_average_of_frequency_in_topic",
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
            "daily_word_occurence_in_topic",
            "daily_word_occurence",
            "total_daily_word_count",
            "topic_daily_word_count",
            "frequency",
            "frequency_in_topic",
            "topic_specificity",
            "five_day_average_of_frequency_in_topic",
            "change_in_average_of_frequency_in_topic",
            "id",
        )

    def run(self):

        word_usage_stats_df = (
            self.comment_tokens_df.select(
                "date",
                "id",
                "topic",
                "word",
            )
            .transform(self.add_comment_id_column)
            .transform(self.add_word_count_columns)
            .transform(self.add_topic_frequency_and_specificity_columns)
            .transform(self.add_five_day_average_of_frequency_in_topic_column)
            .transform(
                self.add_daily_change_in_average_frequency_in_topic_column
            )
            .transform(self.add_id_column)
            .select(
                "id",
                "topic",
                "word",
                "date",
                "daily_word_occurence_in_topic",
                "daily_word_occurence",
                "total_daily_word_count",
                "topic_daily_word_count",
                "frequency",
                "frequency_in_topic",
                "topic_specificity",
                "five_day_average_of_frequency_in_topic",
                "change_in_average_of_frequency_in_topic",
            )
        )

        return self.enforce_schema_and_uniqueness(
            word_usage_stats_df, self.schema
        )

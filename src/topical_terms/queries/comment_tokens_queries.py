from pyspark.ml.feature import StopWordsRemover, Tokenizer
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    broadcast,
    col,
    concat,
    explode,
    from_unixtime,
    lit,
    lower,
    split,
    to_date,
    trim,
)
from pyspark_pipeline.queries import Query


class CommentTokensQuery(Query):
    def __init__(
        self,
        reddit_comments_df: DataFrame,
        subreddit_topics_map_df: DataFrame,
        **kwargs,
    ):
        """
        Query to split reddit comments into cleaned word tokens

        args:
            df: A spark DataFrame containing reddit comments
                along with subreddit information and timestamps
            subreddit_topics_map_df: a mapping between subreddits and topics
                discussed in that subreddit
        """

        self.reddit_comments_df = reddit_comments_df
        self.subreddit_topics_map_df = subreddit_topics_map_df
        super().__init__(**kwargs)

    def add_id_column(self, df: DataFrame) -> DataFrame:
        return df.withColumn(
            "id", concat(col("author"), lit("_"), col("created_utc"))
        ).select(
            "body",
            "id",
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
                "id",
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
        subreddit_topic_map_df = self.subreddit_topics_map_df.withColumn(
            "subreddit", lower(col("subreddit"))
        )

        return df.join(
            broadcast(subreddit_topic_map_df),
            on=["subreddit"],
            how="inner",
        ).select(
            "id",
            "body",
            "date",
            "subreddit",
            "topics",
        )

    def tokenize_comment_body(self, df: DataFrame) -> DataFrame:
        """
        splits the comments into individual words
        """
        tokenizer = Tokenizer(inputCol="body", outputCol="words_token")
        return tokenizer.transform(df).select(
            "id", "topics", "date", "words_token"
        )

    def remove_comment_stop_words(self, df: DataFrame) -> DataFrame:
        """
        Removes stop words like 'and', 'the' etc. from comments
        """

        remover = StopWordsRemover(
            inputCol="words_token", outputCol="words_no_stops"
        )

        return remover.transform(df).select(
            "id",
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
                "id",
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
            .select("id", "topics", "word", "date")
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
                "id",
                "topic",
                "word",
                "date",
            )
        )

    def run(self):

        word_df = (
            self.reddit_comments_df.select(
                "author", "body", "created_utc", "subreddit"
            )
            .transform(self.add_id_column)
            .transform(self.add_date_columns)
            .transform(self.filter_by_eligibility_dates)
            .transform(self.add_topics_column)
            .transform(self.add_word_column)
            .transform(self.explode_topics_column_into_topic_column)
            .select(
                "id",
                "topic",
                "word",
                "date",
            )
        )

        return self.enforce_schema_and_uniqueness(word_df, self.schema)

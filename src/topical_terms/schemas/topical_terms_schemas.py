from pyspark_pipeline.schemas import HiveSchema


class CommentTokensSchema(HiveSchema):
    id = "string"
    date = "string"
    topic = "string"
    word = "string"


class WordStatisticsSchema(CommentTokensSchema):
    change_in_average_of_rate_in_topic = "double"
    topic_daily_word_count = "bigint"
    daily_word_frequency = "bigint"
    daily_word_frequency_in_topic = "bigint"
    rate = "double"
    rate_in_topic = "double"
    five_day_average_of_rate_in_topic = "double"
    topic_specificity = "double"
    total_daily_word_count = "bigint"

from pyspark_pipeline.schemas import HiveSchema


class CommentTokensSchema(HiveSchema):
    id = "string"
    date = "string"
    topic = "string"
    word = "string"


class WordStatisticsSchema(CommentTokensSchema):
    change_in_average_of_frequency_in_topic = "double"
    topic_daily_word_count = "bigint"
    daily_word_occurence = "bigint"
    daily_word_occurence_in_topic = "bigint"
    frequency = "double"
    frequency_in_topic = "double"
    five_day_average_of_frequency_in_topic = "double"
    topic_specificity = "double"
    total_daily_word_count = "bigint"

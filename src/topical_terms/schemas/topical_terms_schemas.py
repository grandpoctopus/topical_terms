from pyspark_pipeline.schemas import EtlSchema


class TopicSpecificTrendingWordsSchema(EtlSchema):
    id = "string"
    topic = "string"
    word = "string"
    date = "string"
    daily_word_occurence_per_topic = "bigint"
    daily_word_occurence = "bigint"
    total_daily_word_count = "bigint"
    daily_topic_word_count = "bigint"
    frequency = "double"
    frequency_in_topic = "double"
    topic_specificity = "double"
    rolling_average_of_daily_frequency = "bigint"
    change_in_rolling_average_of_daily_frequency = "bigint"

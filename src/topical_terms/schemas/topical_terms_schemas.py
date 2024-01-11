from pyspark_pipeline.schemas import EtlSchema


class TopicSpecificTrendingWordsSchema(EtlSchema):
    change_in_rolling_average_of_daily_frequency = "double"
    daily_topic_word_count = "bigint"
    daily_word_occurence = "bigint"
    daily_word_occurence_per_topic = "bigint"
    date = "string"
    frequency = "double"
    frequency_in_topic = "double"
    id = "string"
    rolling_average_of_daily_frequency = "double"
    topic = "string"
    topic_specificity = "double"
    total_daily_word_count = "bigint"
    word = "string"

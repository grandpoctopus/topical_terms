from pyspark_pipeline.schemas import EtlSchema


class TopicSpecificTrendingWordsSchema(EtlSchema):
    change_in_average_of_frequency_in_topic = "double"
    topic_daily_word_count = "bigint"
    daily_word_occurence = "bigint"
    daily_word_occurence_in_topic = "bigint"
    date = "string"
    frequency = "double"
    frequency_in_topic = "double"
    id = "string"
    five_day_average_of_frequency_in_topic = "double"
    topic = "string"
    topic_specificity = "double"
    total_daily_word_count = "bigint"
    word = "string"

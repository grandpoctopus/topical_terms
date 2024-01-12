from pyspark_pipeline.utilities.arg_parse_utils import parse_job_args
from pyspark_pipeline.utilities.job_utils import job_driver
from pyspark_pipeline.utilities.settings_utils import Settings

from topical_terms.jobs.topical_terms_job import TopicalTermsJob


def main():
    job_driver(
        app_name="topical_terms.py",
        arg_parser=parse_job_args,
        job=TopicalTermsJob,
        settings_class=Settings,
    )


if __name__ == "__main__":
    main()

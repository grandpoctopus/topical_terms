FROM datamechanics/spark:3.1.2-hadoop-3.2.0-java-8-scala-2.12-python-3.8-dm17
USER root
COPY /. /opt/topical-terms
RUN cd /opt/topical-terms \
    && conda create -n topical-terms pip python==3.8.13 \
    && source activate topical-terms \
    && conda install -c conda-forge poetry==1.1.14 -y \
    && poetry install

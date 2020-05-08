FROM continuumio/miniconda3

# Set some ENV variables
ENV PATH /usr/sbin:/usr/bin:/sbin:/bin:/opt/conda/envs/snorkel/bin:/opt/conda/condabin:/opt/fasttext

# install packages for installation process
RUN apt-get update && apt-get install unzip build-essential make -y && \
     rm -rf /var/cache/apk/*

# download snorkel
RUN curl -L -o snorkel.zip https://github.com/HazyResearch/snorkel/archive/v0.7.0-beta.zip && \
     unzip snorkel.zip && \
     mv snorkel-0.7.0-beta/ /opt/snorkel && \
     cd /opt/snorkel && \
     # delete unused and huge pytorch package
     sed '/^  - pytorch/d' environment.yml > snorkel.yml && \
     # create snorkel environment
     conda env create --file=snorkel.yml && \
     # install snorkel
     pip install . && \
     python -m ipykernel install --user --name snorkel --display-name "Python (snorkel)" && \
     jupyter nbextension enable --py widgetsnbextension && \
     # install german modul for spacy
     /opt/conda/envs/snorkel/bin/python -c "from snorkel.parser.spacy_parser import Spacy; Spacy(lang='de')" && \
     # install jupyterlab
     pip install jupyterlab && \
     # cleanup
     rm -rf /snorkel.zip /opt/snorkel/test /opt/snorkel/tutorials /opt/snorkel/docs /opt/snorkel/figs

# download fasttext
RUN curl -L -o fasttext.zip https://github.com/facebookresearch/fastText/archive/v0.2.0.zip && \
     unzip fasttext.zip && \
     cd fastText-0.2.0 && \
     # build fasttext
     make && \
     mkdir -p /opt/fasttext && \
     # move the binary to proper path
     mv fasttext /opt/fasttext/fasttext && \
     # cleanup
     rm -rf /fastText-0.2.0 /fasttext.zip


RUN alias python="/opt/conda/envs/snorkel/bin/python"
WORKDIR /opt/project

CMD [ "jupyter", "lab", "--ip", "0.0.0.0", "--allow-root" ]

FROM ingn/tensorflow-gpu-openslide

RUN apt-get install -y vim

ADD .vimrc /root/

WORKDIR /
RUN cd svs_reader &&\
    pip3 install -e .

# ADD tfmodels/* /app/tfmodels/

WORKDIR /
RUN git clone https://github.com/BioImageInformatics/tfmodels &&\
    cd tfmodels &&\
    pip3 install -e .

WORKDIR /
ADD densenet.ckpt-164816* /app/
ADD deploy_trained.py /app/
ADD color_svs.py /app/
ADD densenet/* /app/densenet/

RUN pip3 install scikit-learn &&\
    pip3 install pandas &&\
    pip3 install seaborn

WORKDIR /app

CMD ["/bin/bash"]

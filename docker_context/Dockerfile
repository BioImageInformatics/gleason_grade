FROM ingn/tensorflow:gpu-openslide

RUN apt-get install -y vim

WORKDIR /
RUN ls
RUN cd svs_reader &&\
    pip3 install -e .

ADD tfmodels/* /app/tfmodels/

# WORKDIR /
# RUN git clone https://github.com/nathanin/tfmodels &&\
#     cd tfmodels &&\
#     pip3 install -e .

WORKDIR /
ADD densenet.ckpt-360000* /app/
ADD deploy_trained.py /app/
ADD color_svs.py /app/
ADD densenet/* /app/densenet/

RUN pip3 install scikit-learn &&\
    pip3 install pandas &&\
    pip3 install seaborn

WORKDIR /app

CMD ["/bin/bash"]

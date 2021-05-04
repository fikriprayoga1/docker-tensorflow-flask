FROM ibmcom/tensorflow-ppc64le

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

CMD [ "python", "./app.py" ]
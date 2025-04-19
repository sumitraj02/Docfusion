FROM python:3.10-slim-buster
ENV TZ=Asia/Kolkata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
WORKDIR /volumes

RUN apt update -y && apt install -y \
    ffmpeg libsm6 libxext6 \
    texlive-latex-base texlive-latex-extra texlive-fonts-recommended \
    pandoc
    
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --use-deprecated=legacy-resolver || \
    pip install --no-cache-dir pymilvus grpcio grpcio-status

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


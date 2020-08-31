FROM wandb/deepo:keras AS base
WORKDIR /src
COPY ./data/absolute/2D ./data/absolute/2D
COPY ./sweep_cv.py .
COPY ./data_repository.py .
ENV LC_ALL="C.UTF-8"
ENV LANG="C.UTF-8"
ENV WANDB_PROJECT="slr"
RUN ["wandb", "login"]
ENTRYPOINT ["wandb", "agent", "a4mzdx3k"]

# Build with docker build -t [image_name] .
# Run with docker run -e WANDB_API_KEY=[YOUR_API_KEY] [image_name]
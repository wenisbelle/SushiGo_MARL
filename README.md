# SushiGo_MARL
Requirement for the Multi Agent Systems class

## Requirements 

- Docker

- Nvidia toolkit container

## Environment

Build the image:

    docker build -f dockerfile -t sushi_go .

Create the container:

    docker-compose up -d

    docker exec -it sushi_go_container bash

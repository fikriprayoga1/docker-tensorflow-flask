# Documentation
This is repository of fikriprayoga1/tensorflow-flask in docker hub. The function of this repository is create image of fikriprayoga1/tensorflow-flask in docker hub, so you can learn how to create image and run the docker system image easily. This repository is part of https://hub.docker.com/repository/docker/fikriprayoga1/tensorflow-flask project

## Best Practice
Use this command in your CMD(Windows) or Terminal(Mac or Linux). Before start the command, you must be in the directory of the folder you pulled.

### Step 1. Create fikriprayoga1/tensorflow-flask image
```
docker build -t fikriprayoga1/tensorflow-flask .
```

### Step 2. Create fikriprayoga1/tensorflow-flask container
```
docker container create --name tensorflow-flask -p 5000:5000 fikriprayoga1/tensorflow-flask
```

### Step 3. Run tensorflow-flask container
```
docker start tensorflow-flask
```

### Step 4. Test in your browser
Type in your browser address
```
localhost:5000/testing
```

## Utility Command
This part is command line to help you something

### Show running container list
```
docker container ls
```

### Show image list
```
docker images
```

### Show  container list
```
docker container ls -a
```

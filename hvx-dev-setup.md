```
docker pull --platform=linux/amd64 ubuntu:20.04

docker run --platform=linux/amd64 -it \
  -v "$PWD":/work \
  -w /work \
  ubuntu:20.04 bash
```
```
apt-get update
apt-get install -y \
  build-essential git cmake ninja-build python3 python3-pip \
  wget unzip ca-certificates \
  clang libc++-dev libc++abi-dev
```

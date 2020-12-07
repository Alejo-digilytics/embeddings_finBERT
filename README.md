


 listed in requirements.txt
This can be done moving to the working directory and running the following command on terminal 
`pip install -r requiremnts.txt`


One of the libraries used here is pytorch.
The version depends on the computer and must be compatible with the cuda installed in the computer as well as the OS.
This can be easily done by checking your cuda and downloading the apropiate pytorch version

    1. Check cuda for windows: run the following command in the cmd "nvcc --version"
    2. Check cuda for Linux or Mac: assuming that cat is your editor run "cat /usr/local/cuda/version.txt",
    or the version.txt localization if other

Downloading pytorch: go to `https://pytorch.org/get-started/locally/` and follow the instructions for the download.
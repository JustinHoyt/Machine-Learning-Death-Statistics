sudo yum install -y git
wget https://github.com/git-lfs/git-lfs/releases/download/v2.0.2/git-lfs-linux-amd64-2.0.2.tar.gz
tar -xzf git-lfs-linux-amd64-2.0.2.tar.gz
sudo git-lfs-2.0.2/install.sh 
git clone https://github.com/JustinHoyt/Machine-Learning-Clustering.git
(cd Machine-Learning-Clustering/ && git lfs pull)
wget https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh
chmod 777 Anaconda3-4.3.1-Linux-x86_64.sh
# run this command manually
# bash Anaconda3-4.3.1-Linux-x86_64.sh -p

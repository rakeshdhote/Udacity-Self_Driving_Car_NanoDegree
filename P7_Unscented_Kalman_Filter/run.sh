rm -rf build
mkdir build && cd build
cmake .. && make
# echo "~~~ Data 1 ~~~"
./UnscentedKF ../data/sample-laser-radar-measurement-data-1.txt output1.txt
# echo "~~~ Data 2 ~~~"
./UnscentedKF ../data/sample-laser-radar-measurement-data-2.txt output2.txt
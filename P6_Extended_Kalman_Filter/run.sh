rm -rf build
mkdir build && cd build
cmake .. && make
echo "~~~ Data 1 ~~~"
./ExtendedKF ../data/sample-laser-radar-measurement-data-1.txt output.txt
echo "~~~ Data 2 ~~~"
./ExtendedKF ../data/sample-laser-radar-measurement-data-1.txt output.txt
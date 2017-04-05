# Remove the build 
# to be run once
rm -rf build

# Build project
mkdir build && cd build
cmake .. && make

# Execute code 
echo "~~~ Data 1 ~~~"
./ExtendedKF ../data/sample-laser-radar-measurement-data-1.txt output.txt

echo "~~~ Data 2 ~~~"
./ExtendedKF ../data/sample-laser-radar-measurement-data-2.txt output.txt
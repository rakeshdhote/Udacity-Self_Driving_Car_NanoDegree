rm -rf build
mkdir build && cd build
cmake .. && make

# cd build
# make
echo "~~~~~ RUN ~~~~~~~"
./pid
echo "~~~~~ DONE~~~~~~~"
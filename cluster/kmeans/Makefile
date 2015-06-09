

ver = debug

ifeq ($(ver), debug)
ALL: main
CXXFLAGS = -c -g -Ddebug -std=c++11
out_dir = ./out/debug
else
ALL: main
CXXFLAGS = -c -O3 -std=c++11
out_dir = ./out/release
endif

main : main.o owbalgo_kmeans.o
	g++ -o $(out_dir)/$@ $(out_dir)/main.o $(out_dir)/owbalgo_kmeans.o

%.o: %.cpp
	g++ $(CXXFLAGS) $< -o $(out_dir)/$@


.PHONY : clean
clean :
	-rm -f $(out_dir)/main $(out_dir)/%.o

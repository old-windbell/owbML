

ver = debug

ifeq ($(ver), debug)
ALL: main
CXXFLAGS = -c -g -Ddebug
out_dir = ./out/debug
else
ALL: main
CXXFLAGS = -c -O3
out_dir = ./out/release
endif

main : main.o owbalgo_svm.o
	g++ -o $(out_dir)/$@ $(out_dir)/main.o $(out_dir)/owbalgo_svm.o

%.o: %.cpp
	g++ $(CXXFLAGS) $< -o $(out_dir)/$@


.PHONY : clean
clean :
	-rm -f $(out_dir)/main $(out_dir)/%.o

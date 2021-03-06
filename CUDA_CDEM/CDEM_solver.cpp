// CDEM_solver.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include "Domain.h"


int main(int argc, char** argv)
{
	//Domain dom = Domain();
	//dom.load_from_file("D:/Dropbox/VladimirVancik/CUDA/jobs/3pb_large_CUDA/3pb.txt");
	//dom.solve(0.03, 0.03, 100000, "D:/Dropbox/VladimirVancik/CUDA/jobs/3pb_large_CUDA/3pbo",5000);
	//std::cout << dom.nodes[58].v_disp << std::endl;
	//std::cin.get();
	std::string f;
	char * o;
	FLOAT_TYPE t1, t2;
	int ns, of, gd, bd;
	if (argc == 2)
	{
		std::cout << "This is a simple solver for FEM domain of elements connected via springs." << std::endl
			<< "Following is a list of arguments, all of which must be given" << std::endl << std::endl;
		std::cout << "-f\tSpecifies input file path." << std::endl;
		std::cout << "-o\tSpecifies output file path." << std::endl;
		std::cout << "-t1\tTime - load function reaches 1.0." << std::endl;
		std::cout << "-t2\tTime - maximum time." << std::endl;
		std::cout << "-ns\tNumber of steps." << std::endl;
		std::cout << "-of\tOutput frequency." << std::endl;
		std::cout << "-gd\tGrid dimension." << std::endl;
		std::cout << "-bd\tBlock dimension." << std::endl;
	}
	else if (argc == 17){
		for (int i = 1; i < 17; i += 2)
		{
			if (std::string(argv[i]) == "-f") {
				f = argv[i + 1];
			}
			else if (std::string(argv[i]) == "-o") {
				o = argv[i + 1];
			}
			else if (std::string(argv[i]) == "-t1") {
				t1 = std::stod(argv[i + 1]);
			}
			else if (std::string(argv[i]) == "-t2") {
				t2 = std::stod(argv[i + 1]);
			}
			else if (std::string(argv[i]) == "-ns") {
				ns = std::stoi(argv[i + 1]);
			}
			else if (std::string(argv[i]) == "-of") {
				of = std::stoi(argv[i + 1]);
			}
			else if (std::string(argv[i]) == "-gd") {
				gd = std::stoi(argv[i + 1]);
			}
			else if (std::string(argv[i]) == "-bd") {
				bd = std::stoi(argv[i + 1]);
			}
			else {
				std::cout << "Unknown argument " << argv[i] << ", calculation cancelled." << std::endl;
				return 1;
			}
		}
		Domain dom = Domain();
		dom.load_from_file(f);
		dom.solve(t1, t2, ns, o, of, gd, bd);
	}
	else {
		//std::cout << "Wrong number of arguments." << std::endl;
		Domain dom = Domain();
		dom.load_from_file("D:/Dropbox/VladimirVancik/CUDA/jobs/3pb_large_CUDA/3pb.txt");
		dom.solve(0.03, 0.03, 100000, "D:/Dropbox/VladimirVancik/CUDA/jobs/3pb_large_CUDA/3pbo",5000,1,512);
		//std::cout << dom.nodes[2998].v_disp << std::endl;
	}
	std::cout << "Calculation complete." << std::endl;
	std::cin.get();
	return 0;
}
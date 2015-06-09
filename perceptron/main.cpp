#include "../base/owbdata.h"
#include "../base/owbmodel.h"
#include "owbalgo_perceptron.h"

int main(int argc, char* argv[])
{
	using std::cout;
	using std::string;
	
	string data_file("../../train.data");
	ParaPerceptron para;
	if (argc >= 2) {
		para.type = PerceptronType(std::atoi(argv[1]));
	}
	else if (argc >= 3) {
		data_file = argv[2];
	}

	int32 ret = 0;
	
    owbDataValue data;
    ret = data.load_data(data_file.c_str());
    if (ret < 0) {
		if (ret == ERR_OPEN) {
			cout<<"open file error!\n";
		}
		
		return ret;
	}

    data.print_data();

    owbAlgoPeceptron algo;
    algo.set_parameter(para);
    algo.train(&data);

    owbModel *model = algo.get_model();
    model->print();
    
    return 0;
}

#include "../base/owbdata.h"
#include "../base/owbmodel.h"
#include "owbalgo_svm.h"


int main(int argc, char* argv[])
{
    owbDataValue data;
    data.load_data("../../train.data");

    data.print_data();

    owbAlgoSVM svm;
    svm.train(&data);

    owbModel *model = svm.get_model();
    model->print();
    
    return 0;
}

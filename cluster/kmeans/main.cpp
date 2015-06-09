#include "../../base/owbinclude.h"
#include "owbalgo_kmeans.h"

int main(int argc, char* argv[])
{
    owbDataValue data;
    data.load_data("../../train.data");

    data.print_data();

    owbAlgoKMeans kmeans;
    kmeans.train(&data);

    owbModel *model = kmeans.get_model();
    model->print();
    
    return 0;
}

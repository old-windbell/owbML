#include "owbalgo_apriori.h"

int main(int argc, char* argv[])
{
	owbDataValue data;
	data.load_data("../../train.data");

//	data.print_data();

	owbAlgoApriori algo;
	algo.train(&data);

	owbModelApriori *model = dynamic_cast<owbModelApriori*>(algo.get_model());
	model->print();

	return 1;
}

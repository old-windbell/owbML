#ifndef OWB_ALGO_SVM
#define OWB_ALGO_SVM

#include "../base/owbinclude.h"

class owbDataValue;

//Kernal type
enum KernalType
{
    POLYNOMIAL = 0,
};

//svm parameters
struct ParaSVM
{
    double C;            //penalty for miss classify
    double tolerance;    //calculate accuracy
    KernalType kernal;   //kernal type
    uint32 count;        //calculate count
    double delta;        //minmum step
};


//svm model
class owbModelSVM : public owbModel
{
 public:
    std::vector<double> a;
    double b;

    void init(uint32 dim) {
		a.resize(dim);
		b = 0;
    }

    virtual void print() {
	std::cout<< "a: ";
	for (uint32 i = 0; i< a.size(); ++i) {
	    std::cout<< a[i] <<',';
	}

	std::cout<< '\n' << "b: "<< b <<'\n';
    }
};

//@class  : owbAlgoSVM
//@func   : svm algorithm
class owbAlgoSVM : public owbAlgo
{
 public:
    //train data
    virtual int32 train(owbData* data);

    //get model
    virtual owbModel* get_model();

    //constructor
    owbAlgoSVM();        //default parameter
    owbAlgoSVM(const ParaSVM* para);
    
    //set svm parameter
    void set_para(const ParaSVM* para);

    virtual ~owbAlgoSVM(){}

 protected:
    //kernal function, calculate kernal value
    double kernal(uint32 j, uint32 i);

    //polynomial kernal
    double kernal_polynomial(uint32 j, uint32 i);

    // predict real diffrence
    double predict_real_diff(uint32 i);

    // init e dict
    void init_e_dict();

    // update error dict
    void update_e_dict();
    
 private:
    ParaSVM m_para;         //parameter
    owbModelSVM m_model;    //result model
    owbDataValue *m_data;   //train data

    std::vector<double> m_e_dict;//error dictionary
    
};


#endif //OWB_ALGO_SVM

#ifndef OWB_ALGO_PERCEPTRON
#define OWB_ALGO_PERCEPTRON

#include "../base/owbinclude.h"

// perceptron type
enum PerceptronType
{
	SYMMETRY		= 0,
	RAW_PERCEPTRON	= 1,
};


//parameters
struct ParaPerceptron
{
	double eta;
	PerceptronType type;
	
	ParaPerceptron() : eta(1), type(SYMMETRY){}
};


//perceptron model
class owbModelPerceptron : public owbModel
{
public:
	itemv alpha;
	double b;

public:
	void init(uint32 count, double _b = 0) {
		alpha.resize(count);
		b = _b;
	}

	virtual void print() {
		std::cout<<"alpha: ";
		uint32 sz = alpha.size();
		for (uint32 i = 0; i < sz; ++i) {
			std::cout<<i <<":" <<alpha[i] <<' ';
		}
		
		std::cout<<'\n';
		
		std::cout<<"b: " <<b <<'\n';
	}
};


// perceptron algorithm
class owbAlgoPeceptron : public owbAlgo
{
public:
	//train data
    virtual int32 train(owbData* data);

    //get model
    virtual owbModel* get_model();
    
public:
	//perceptron symmetry training algorithm
	int32 train_symmetry(owbDataValue *data);
	
	//raw perceptron
	int32 train_raw_perceptron(owbDataValue *data);
	
	//set parameter
	void set_parameter(const ParaPerceptron& para);
	
    
protected:
	//calculate the Gram matrix to accelerate training
	int32 calc_gram_matrix(owbDataValue* data);
    
private:
	ParaPerceptron m_para;
	owbModelPerceptron m_model;
	
	std::vector<itemv*> m_gram_matrix;
};


#endif //OWB_ALGO_PERCEPTRON

#ifndef OWB_ALGO_KMEANS
#define OWB_ALGO_KMEANS

#include "../../base/owbinclude.h"

struct ParaKMeans
{
    uint32 k;
};

class owbModelKMeans : public owbModel
{
 public:
    virtual void print() {
        using std::cout;

        cout<<"Show the model-- cluster  centers:\n";

    	uint32 cnt = centers.size();
    	for (uint32 i = 0; i < cnt; ++i) {
            cout<<i <<": ";
            for (uint32 j = 0; j < centers[i]->size(); ++j) {
                cout<< centers[i]->at(j) << ' ';
            }

            cout<<'\n';
	    }
    }

 public:
    std::vector<itemv*> centers;
};


class owbAlgoKMeans : public owbAlgo
{
 public:
    //train data
    virtual int32 train(owbData* data);

    //get model
    virtual owbModel* get_model();

 public:
    void init(owbData* data);

    owbAlgoKMeans(uint32 k = 2);
    ~owbAlgoKMeans();
    
 private:
    owbModelKMeans m_model;
    ParaKMeans  m_para;
};

#endif //OWB_ALGO_KMEANS

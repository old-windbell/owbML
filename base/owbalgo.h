#ifndef OWB_ALGO
#define OWB_ALGO

#include "owbdefs.h"

class owbData;
class owbModel;

//@class  : owbAlgo
//@func   : all algorithms interface
class owbAlgo
{
 public:
    virtual ~owbAlgo() {
	}

    //@info train the data
    //@para the data to train
    //@ret  if success, return 0, else not 0
    virtual int32 train(owbData* data) = 0;

    //get model of training result
    virtual owbModel* get_model() = 0;

};

#endif //OWB_ALGO

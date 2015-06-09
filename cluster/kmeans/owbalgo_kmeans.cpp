#include "owbalgo_kmeans.h"
#include <climits>


owbAlgoKMeans::owbAlgoKMeans(uint32 k)
{
    m_para.k = k;
}

owbAlgoKMeans::~owbAlgoKMeans()
{
}

void owbAlgoKMeans::init(owbData* data)
{
    
}

int32 owbAlgoKMeans::train(owbData* data)
{
    owbDataValue* datav = (owbDataValue*)data;
    std::vector<uint32> lbs;  //labels
    lbs.resize(data->get_count());

    uint32 i = 0, data_dim = datav->get_dim();
    for (i = 0; i < m_para.k; ++i) {
        itemv* ptem = new itemv();
        ptem->resize(data_dim);
        memset(ptem->data(), 0, sizeof(double) * data_dim);

	    if (i < data->get_count()) {
	        *ptem = *(datav->get_item(i));
	    }

        
    	m_model.centers.push_back(ptem);
    }

    while(true) {
        bool change_flag = false;
        for (i = 0; i < datav->get_count(); ++i) {
            uint32 min_dis = UINT_MAX, cls_flag = 0;
            for (uint32 j = 0; j < m_para.k; ++j ) {
                double dis = datav->calc_dot_product(m_model.centers[j], datav->get_item(i));
                if (dis < min_dis) {
                    min_dis = dis;
                    cls_flag = j;
                }
            }

            if (lbs[i] != cls_flag) {
                change_flag = true;
                lbs[i] = cls_flag;
            }
        }

        if (!change_flag) {
            break;
        }
    }

    return 0;
}

owbModel* owbAlgoKMeans::get_model()
{
    return &m_model;
}

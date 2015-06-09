#include "owbalgo_svm.h"

#define fabs(a) ((a)>=0?(a):-(a))

static uint32 pc = 0;

owbAlgoSVM::owbAlgoSVM()
{
    m_para.C = 10.0;
    m_para.tolerance = 0.0001;
    m_para.kernal = POLYNOMIAL;
    m_para.count = 100;
    m_para.delta = 0.001;
}

owbAlgoSVM::owbAlgoSVM(const ParaSVM* para)
{
    //memcpy(&m_para, para, sizeof(m_para));
    m_para = *para;
}

//set svm parameter
void owbAlgoSVM::set_para(const ParaSVM* para)
{
	m_para = *para;
}

int32 owbAlgoSVM::train(owbData* data)
{
    m_data = dynamic_cast<owbDataValue*>(data);
    if (m_data == NULL)
	return -1;
    
    m_model.a.resize(m_data->get_count());
	
    uint32 count = 0;
    init_e_dict();
    bool updated = true;

    while (count < m_para.count && updated) {
	updated = false;
	count += 1;

	for (uint32 i = 0; i < m_model.a.size(); ++i) {
	    double ai = m_model.a[i];
	    double Ei = m_e_dict[i];

	    // against the KKT
	    if ((m_data->get_label(i) * Ei < -m_para.tolerance && ai < m_para.C)
		|| (m_data->get_label(i) * Ei > m_para.tolerance && ai > 0)) {
		for (uint32 j = 0; j < m_model.a.size(); j++) {
		    if (j == i)
			continue;
		    double eta = kernal(j, j) + kernal(i, i) - kernal(i, j) * 2;
		    if (eta <= 0)
			continue;
		    double new_aj = m_model.a[j] + m_data->get_label(j) *
			(m_e_dict[i] - m_e_dict[j]) / eta;
		    
		    double L = 0, H = 0;
		    if (m_data->get_label(i) == m_data->get_label(j)) {
			L =std::max(0.0, m_model.a[j] + m_model.a[i] - m_para.C);
			H = std::min(m_para.C, m_model.a[j] + m_model.a[i]);
		    } else {
			L = std::max(0.0, m_model.a[j] - m_model.a[i]);
			H = std::min(m_para.C, m_para.C + m_model.a[j] - m_model.a[i]);
		    }

		    if (new_aj > H)
			new_aj = H;
		    if (new_aj < L)
			new_aj = L;

		    double new_ai = m_model.a[i] + m_data->get_label(i) *
			m_data->get_label(j) * (m_model.a[j] - new_aj);

		    //decline enough for new_aj
		    if (fabs(m_model.a[j] - new_aj) < m_para.delta)
			continue;

		    double new_b1 = m_model.b - m_e_dict[i] - m_data->get_label(i) *
			kernal(i, i) * (new_ai - m_model.a[i]) -
			m_data->get_label(j) * kernal(j, i) * (new_aj - m_model.a[j]);
		    double new_b2 = m_model.b - m_e_dict[j] - m_data->get_label(i) *
			kernal(i, j) * (new_ai - m_model.a[i]) -
			m_data->get_label(j) * kernal(j, j) * (new_aj - m_model.a[j]);

		    double new_b = 0;
		    if (new_ai > 0 && new_ai < m_para.C)
			new_b = new_b1;
		    else if (new_aj > 0 && new_aj < m_para.C)
			new_b = new_b2;
		    else
			new_b = (new_b1 + new_b2) / 2.0;

		    m_model.a[i] = new_ai;
		    m_model.a[j] = new_aj;
		    m_model.b = new_b;
		    //std::cout<<pc++ <<"::i=" <<i <<" j=" <<j <<"  ";
		    //std::cout<< "ai="<<new_ai <<" aj=" <<new_aj <<"b=" <<new_b <<'\n';
		    update_e_dict();
		    updated = true;
		}
	    }
	}
    }
    
    return 0;
}

owbModel* owbAlgoSVM::get_model()
{
    return &m_model;
}

double owbAlgoSVM::kernal(uint32 j, uint32 i)
{
    if (m_para.kernal == POLYNOMIAL){
	return kernal_polynomial(j, i);
    }

    return 0;
}

double owbAlgoSVM::kernal_polynomial(uint32 j, uint32 i)
{
    bool err = false;
    itemv* item_j = m_data->get_item(j);
    itemv* item_i = m_data->get_item(i);
    if (item_j == NULL || item_i == NULL) {
		return 0;
	}
	
    double ret = 0;
    uint32 dim = m_data->get_dim();
    for (uint32 idx = 0; idx < dim; ++idx) {
	ret += item_j->at(idx) * item_i->at(idx);
    }
    
    return ret;
}

double owbAlgoSVM::predict_real_diff(uint32 i)
{
    double diff = 0;
    for (uint32 j = 0; j < m_data->get_count(); ++j) {
	diff += m_model.a[j] * m_data->get_label(j) * kernal(j, i);
    }

    diff = diff + m_model.b - m_data->get_label(i);

    return diff;
}

void owbAlgoSVM::init_e_dict()
{
    if (!m_data)
	return;
    
    m_e_dict.resize(m_data->get_count());
    update_e_dict();
}

void owbAlgoSVM::update_e_dict()
{
    for (int i = 0; i < m_e_dict.size(); ++i) {
	m_e_dict[i] = predict_real_diff(i);
    }
}

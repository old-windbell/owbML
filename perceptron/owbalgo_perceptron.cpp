#include "owbalgo_perceptron.h"

int32 owbAlgoPeceptron::train(owbData* data)
{
	owbDataValue* datav = dynamic_cast<owbDataValue*>(data);
	if (datav == NULL) {
		return -1;
	}
	
	int ret = 0;
	if (m_para.type == SYMMETRY) {
		m_model.init(datav->get_count(), 1);
		ret = train_symmetry(datav);
	}
	else if (m_para.type == RAW_PERCEPTRON) {
		m_model.init(datav->get_dim(), 0);
		ret = train_raw_perceptron(datav);
	}
	
	return ret;
}

//@func : perceptron symmetry training algorithm
//@para : training data
//@retn : if success, return 0, else less than 0
int32 owbAlgoPeceptron::train_symmetry(owbDataValue* data)
{
	int32 ret = calc_gram_matrix(data);
	if (ret < 0) {
		return ret;
	}
	
	uint32 cnt = data->get_count();
	while(true) {
		bool no_change = true;
		
		for (uint32 i = 0; i < cnt; ++i) {
			double rst = m_model.b;	//result
			for (uint32 j = 0; j < cnt; ++j) {
				rst += m_model.alpha[j] * data->get_label(j) * 
					data->calc_dot_product(i, j);
			}
			rst *= data->get_label(i);
			
			if (rst <= 0) {
				m_model.alpha[i] += m_para.eta;
				m_model.b += m_para.eta * data->get_label(i);
				
				no_change = false;
			}
		}
		
		if (no_change) {
			break;
		}
	}
	
	return ret;
}

//@func : perceptron raw training algorithm
//@para : training data
//@retn : if success, return 0, else less than 0
int32 owbAlgoPeceptron::train_raw_perceptron(owbDataValue* data)
{
	while (true) {
		bool no_change = true;
		for (uint32 i = 0; i < data->get_count(); ++i) {
			double dp = data->calc_dot_product(&(m_model.alpha), data->get_item(i), true);
			double label = data->get_label(i);
			double judge = label * (dp + m_model.b); 
			
			if (judge <= 0) {
				uint32 dim = m_model.alpha.size();
				for (uint32 j = 0; j < dim; ++j) {
					m_model.alpha[j] += m_para.eta * label * data->get_data(i, j);
				}
				
				m_model.b += m_para.eta * label;
#ifdef debug
				m_model.print();
#endif	
				no_change = false;
			}
		}
		
		if (no_change) {
			break;
		}
	}
	
	return 0;
}

void owbAlgoPeceptron::set_parameter(const ParaPerceptron& para)
{
	m_para = para;
}

owbModel* owbAlgoPeceptron::get_model()
{
	return &m_model;
}

//@func : calculate the Gram matrix
//@para : training data
//@retn : if success, return 0, else less than 0
int32 owbAlgoPeceptron::calc_gram_matrix(owbDataValue* data)
{
	uint32 cnt = data->get_count();
	m_gram_matrix.resize(cnt);
	
	uint32 i = 0, j = 0;
	for (i = 0; i < cnt; ++i) {
		m_gram_matrix[i] = new(std::nothrow) itemv;
		if (m_gram_matrix[i] == NULL) {
			return -1;
		}
		
		m_gram_matrix[i]->resize(cnt);
		
		for (j = i; j < cnt; ++j) {
			m_gram_matrix[i]->at(j) = data->calc_dot_product(i, j);
		}
	}
	
	for (i = 1; i < cnt; ++i) {
		for (j = 0; j < i; ++j) {
			m_gram_matrix[i]->at(j) = m_gram_matrix[j]->at(i);
		}
	}
	
	return 0;
}

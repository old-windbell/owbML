//@file   : owbData
//@author : old-windbell
//@func   : for dealing data, loading...

#ifndef OWB_DATA
#define OWB_DATA

#include "owbdefs.h"
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

using std::string;


//@class  : owbData
//@func   : data base class
class owbData
{
 public:
    //load data from a data file
    virtual int32 load_data(const char* file) = 0;

    //print data
    virtual void print_data() = 0;

    virtual uint32 get_count() const = 0;

    ~owbData(){}

 protected:
    // string split
    void split(const string &src, char token, std::vector<string> &vect) {
		string::size_type pos = 0, nbegin = 0;
		while (pos != string::npos){
	    	pos = src.find_first_of(token, nbegin);
	    	if (pos == string::npos){
				vect.push_back(src.substr(nbegin));
				break;
	    	} else {
				vect.push_back(src.substr(nbegin, pos-nbegin));
	    	}
	    	nbegin = pos + 1;
		}
    }

    double stod(const char* str) {
		std::istringstream iss(str);
		double num = 0;
		iss>> num;
		return num;
    }
};

// item define
typedef std::vector<double> itemv;    //value item

const uint32 BUF_SIZE = 1024;

//@class  : owbDataValue
//@func   : for data only with number, no text
class owbDataValue : public owbData
{
public:
	~owbDataValue() {
		uint32 cnt = m_datas.size();
		for (uint32 i = 0; i < cnt; ++i) {
			delete m_datas[i];
		}
	}
	
    virtual int32 load_data(const char* filename) {
		std::fstream f(filename);
		if (!f.is_open()) {
		    return ERR_OPEN;
		}

		// 
		char buf[BUF_SIZE] = {0};
		while(f.getline(buf, BUF_SIZE)) {
			if (buf[0] == '#') {
				continue;
			}

	   		std::vector<std::string> vect;
	    	split(buf, ',', vect);

			itemv* new_item = new itemv;
	    	std::vector<string>::iterator itr = vect.begin();
	    	for (; itr != vect.end(); ++itr){
				new_item->push_back(stod(itr->c_str()));
	    	}

	   		m_datas.push_back(new_item);
		}

		f.close();	
    }

    // print data
    virtual void print_data() {
		uint32 count = m_datas.size();

		itemv::iterator itr;
		for (int i = 0; i < count; ++i ) {
	    	for (itr = m_datas[i]->begin(); itr != m_datas[i]->end(); ++itr) {
				std::cout<< *itr<< ",";
			}

	    	std::cout<<"\n";
		}
    }

    // get item count
    virtual uint32 get_count() const {
		return m_datas.size();
    }

    //get label
    double get_label(uint32 item) {
		if (!m_datas[item]->empty()) {
	    	return m_datas[item]->back();
		}

		return 0;
    }
    
    // get a value in the matrix
    double get_data(uint32 i, uint32 j) const {
		if (i <= m_datas.size()) {
	    	if (j <= m_datas[i]->size()) {
				return m_datas[i]->at(j);
			}
		}

		return 0;
    }

    // get item
    itemv* get_item(uint32 i) const {
		if (i <= m_datas.size()) {
	    	return m_datas[i];
		}

		return NULL;
    }

    // get dimention
    uint32 get_dim() const {
		return m_datas[0]->size() - 1;
    }
    
    // dot product
    double calc_dot_product(uint32 id0, uint32 id1, bool f = false) const {
		if (id0 >= m_datas.size() || id1 >= m_datas.size()) {
			return 0;
		}
		
		return calc_dot_product(m_datas[id0], m_datas[id1], f);
	}
	
	// dot product
	// para : if the parameter f is false, the last data(often present 
	//         the label) in itemv will not be calculated
	double calc_dot_product(const itemv* v1, const itemv* v2, bool f = false) const {
		if (v1 == NULL || v2 == NULL) {
			return 0;
		}
		
		uint32 dim = std::min(v1->size(), v2->size());
		if (!f) {
			dim -= 1;
		}
		
		double rst = 0;
		for (uint32 i = 0; i < dim; ++i) {
			rst += v1->at(i) * v2->at(i);
		}
		
		return rst;
	}
	
	// calculate a real number mutliply a vector
	itemv real_multiply_vector(double real, const uint32 _id, bool f = false) const {
		return real_multiply_vector(real, m_datas[_id], f);
	}
	
	// calculate a real number mutliply a vector
	itemv real_multiply_vector(double real, const itemv* vct, bool f = false) const {
		itemv _item;
		if (vct == NULL) {
			return _item;
		}
		
		_item = *vct;
		uint32 dim = f ? vct->size() : vct->size() - 1;
		for (uint32 i = 0; i < dim; ++i) {
			_item[i] *= real;
		}
		
		return _item;
	}

private:
    std::vector<itemv*> m_datas;
};


#endif

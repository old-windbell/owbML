#include "owbalgo_apriori.h"
#include <set>
#include <map>
#include <cassert>

owbAlgoApriori::owbAlgoApriori() : m_min_sup(2) {
}

owbAlgoApriori::~owbAlgoApriori() {
	size_t lks_sz = m_model.m_Lks.size();
	for (size_t i = 0; i < lks_sz; ++i) {
		itemset_array* parray = m_model.m_Lks[i];
		if (parray == NULL) {
			continue;
		}

		size_t ary_sz = parray->size();
		for (size_t j = 0; j < ary_sz; ++j) {
			itemset* pset = parray->at(j);
			if(pset == NULL) {
				continue;
			}

			delete pset;
			pset  = NULL;
		}

		delete parray;
		parray = NULL;
	}

	m_model.m_Lks.clear();
}

/*
void print_sets(const itemset_array& isa, owbModelApriori &mdl) {
	for (size_t i = 0; i < isa.size(); ++i) {
		std::cout<<"{ ";
		const itemset &tem = *isa[i];
		for (size_t j = 0; j < tem.size(); ++j ) {
			std::cout<< tem[j] <<',';
		}
		std::cout<<"} count: " << mdl.m_item_count[&tem] <<"\n";
	}
}
*/

int32 owbAlgoApriori::train( owbData* data ) {
	itemset_array *pL1 = new(std::nothrow) itemset_array;
	if (pL1 == NULL) {
		return ERR_NULLPTR;
	}

	find_frequent_1_itemsets(data, *pL1);

	owbDataValue *pDataV = dynamic_cast<owbDataValue*>(data);
	if (pDataV == NULL) {
		return ERR_NULLPTR;
	}

	Lks_t& lks = m_model.m_Lks;
	lks.push_back(pL1);

	for (size_t k = 1; lks[k-1]->size() != 0; ++k ) {
		itemset_array *pLpre = lks[k-1];
		itemset_array *pCk = new(std::nothrow) itemset_array;
		if (pCk == NULL) {
			return ERR_NULLPTR;
		}

		lks.push_back(pCk);

		apriori_gen(*pLpre, *pCk);

		item_count_t &item_cnt = m_model.m_item_count;

		uint32 data_cnt = pDataV->get_count();
		for (size_t dindex = 0; dindex < data_cnt; ++dindex) {
			itemv *pitem = pDataV->get_item(dindex);

			for (size_t cindex = 0; cindex < pCk->size(); ++cindex) {
				if (is_subset(*pitem, *pCk->at(cindex))) {
					itemset *pkey = pCk->at(cindex);
					if (item_cnt.find(pkey) == item_cnt.end()) {
						item_cnt[pkey] = 1;
					} else {
						item_cnt[pkey] += 1;
					}
				}
			}
		}

		// delete itemset which count less than min support count
		for (size_t i = 0; i < pCk->size();) {
			itemset* pkey = pCk->at(i);
			if (item_cnt[pkey] < m_min_sup) {
				delete pkey;
				itemset_array::iterator it = pCk->begin();
				it += i;
				pCk->erase(it);
				item_cnt.erase(pkey);
			} else {
				i++;
			}
		}
	}

	return 0;
}

void owbAlgoApriori::find_frequent_1_itemsets(owbData* data,
											itemset_array& L1 ) {
	owbDataValue* pdata = dynamic_cast<owbDataValue*>(data);
	if (pdata == NULL) {
		return;
	}

	typedef std::map<double, uint32> f1_map_t;
	f1_map_t f1_map;

	uint32 item_count = data->get_count();
	for (uint32 i = 0; i < item_count; ++i) {
		itemv* pitem = pdata->get_item(i);
		if (pitem == NULL) {
			continue;
		}

		uint32 trans_count = pitem->size();
		for (uint32 j = 1; j < trans_count; ++j) {
			double val = (*pitem)[j];
			f1_map_t::iterator it = f1_map.find(val);
			if (it != f1_map.end()) {
				f1_map[val] += 1;
			} else {
				f1_map[val] = 1;
			}
		}
	}

	f1_map_t::iterator it = f1_map.begin();
	for (; it != f1_map.end(); ++it) {
		if (it->second >= m_min_sup ) {
			itemset *pitem = new itemset;
			if (pitem == NULL) {
				continue;
			}
			pitem->push_back(it->first);
			L1.push_back(pitem);

			item_count_t &icount = m_model.m_item_count;
			icount[pitem] = it->second;
		}
	}
	std::cout<<'\n';
}

void owbAlgoApriori::apriori_gen(const itemset_array& Lpre, itemset_array& Lk) {
	if (Lpre.size() == 0) {
		return;
	}

	size_t pre_sz = Lpre.size();
	for (size_t i = 0; i < pre_sz; ++i) {
		for (size_t j = i + 1; j < pre_sz; ++j) {
			if (can_merge(*Lpre[i], *Lpre[j])) {
				itemset *pset = new(std::nothrow) itemset(*Lpre[i]);
				size_t j_index = Lpre[j]->size() - 1;
				pset->push_back(Lpre[j]->at(j_index));

				if (!has_infrequent_subset(*pset, Lpre)) {
					Lk.push_back(pset);
				}
			}
		}
	}
}

bool owbAlgoApriori::has_infrequent_subset(const itemset& c,
										const itemset_array& Lpre) {
	size_t sz = c.size();
	for (size_t i = 0; i < sz; ++i) {
		itemset subset(sz - 1);
		size_t id = 0;
		for (size_t j = 0; j < sz; j++) {
			if (j != i) {
				subset[id++] = c[j];
			}
		}

		// if the subset in subset array
		bool not_in = true;
		size_t array_sz = Lpre.size();
		for (size_t i = 0; i < array_sz; ++i) {
			if (*Lpre[i] == subset) {
				not_in = false;
				break;
			}
		}

		if (not_in) {
			return true;
		}
	}

	return false;
}

bool owbAlgoApriori::can_merge(const itemset& lhs, const itemset& rhs) {
	assert(lhs.size() == rhs.size());

	size_t sz = lhs.size() - 1;
	for (size_t i = 0; i < sz; ++i) {
		if (lhs[i] != rhs[i]) {
			return false;
		}
	}

	return true;
}


bool owbAlgoApriori::is_subset(const itemset& sup_set, 
								const itemset& cld_set) const {
	size_t sup_sz = sup_set.size();
	size_t cld_sz = cld_set.size();
	if (sup_sz - 1 < cld_sz) {
		return false;
	}

	size_t sup_index = 1;
	for (size_t i = 0; i < cld_sz; ) {
		if (i >= cld_sz) {
			return true;
		}

		if (sup_index >= sup_sz) {
			return false;
		}

		if (sup_set[sup_index] == cld_set[i]) {
			sup_index += 1;
			i += 1;
		} else if (sup_set[sup_index] < cld_set[i]) {
			sup_index += 1;
		} else {
			return false;
		}
	}

	return true;
}

void owbModelApriori::print() {
	Lks_t::size_type sz = m_Lks.size(), i = 0;
	for (; i < sz; ++i) {
		print_itemset_array(*m_Lks[i]);
	}
}

void owbModelApriori::print_itemset(const itemset& iset) {
	itemset::size_type sz = iset.size(), i = 0;
	std::printf("{");
	for (; i < sz; ++i) {
		std::printf("%.0f,", iset[i]);
	}
	const itemset* pkey = &iset;
	std::printf("} : %d\n", m_item_count[pkey]);
}

void owbModelApriori::print_itemset_array(
		const itemset_array& iarray) {
	typedef itemset_array::size_type sz_t;
	for (sz_t i = 0; i < iarray.size(); ++i) {
		itemset *pset = iarray[i];
		print_itemset(*pset);
	}
}

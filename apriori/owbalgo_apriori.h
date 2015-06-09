//@file		: owbalgo_apriori.h
//@author	: owb	
//@func		: implemet base Apriori algorithm

#ifndef OWBALGO_APRIORI_H
#define OWBALGO_APRIORI_H

#include "../base/owbinclude.h"
#include <map>

typedef std::vector<double> itemset;
typedef std::vector<itemset*> itemset_array;
typedef std::vector<itemset_array*> Lks_t;
typedef std::map<const itemset*, uint32> item_count_t;

class owbModelApriori : public owbModel
{
public:
	Lks_t m_Lks;
	item_count_t m_item_count;

	virtual void print();

private:
	void print_itemset(const itemset& iset);
	void print_itemset_array(const itemset_array& iarray);
};

class owbAlgoApriori : public owbAlgo
{
public:
 	virtual int32 train(owbData* data);

	virtual owbModel* get_model(){
		return &m_model;
	};

	owbAlgoApriori();
	
	virtual ~owbAlgoApriori();

public:
	void find_frequent_1_itemsets(owbData* data, itemset_array& L1);

	void apriori_gen(const itemset_array& Lpre, itemset_array& Lk);

	bool has_infrequent_subset(const itemset& c, const itemset_array& Lpre);

private:
	bool can_merge(const itemset& lhs, const itemset &rhs);

	bool is_subset(const itemset& sup_set, const itemset& cld_set) const;

private:
	uint32 m_min_sup;	// minimum support
	owbModelApriori m_model;
};

#endif //OWBALGO_APRIORI_H




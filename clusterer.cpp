// %%% Clusterer %%%

#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <stdlib.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <sys/time.h>

#define MAX_CATEGORIES	15

#define NUM_FEATURES	4

#define LEARN_SAMPLES	50

using namespace std;

typedef struct
{
    int numCategories = 0;
    int maxNumCategories = 100;
    vector< vector<float> > weight;
    float vigilance = 0.955;
    float bias = 0.01;
    int numEpochs = 100;
    float learningRate = 0.1;

}art_network;

// clustering_mat = zeros(6,MAX_CATEGORIES); //%error matrix for max of MAX_CATEGORIES groups
// clusterd_pdw = cell(MAX_CATEGORIES,1);

art_network ART_Create_Network()
{
	vector< vector<float> > weights;

	art_network net;

	vector<float> init_weight(NUM_FEATURES*2,1.0);

	weights.push_back(init_weight);

	net.numCategories=0;
	net.maxNumCategories = 100;
	net.vigilance = 0.955;
	net.bias = 0.01;
	net.numEpochs = 100;
	net.learningRate = 0.1;
	net.weight = weights;

	return net;
}

void normalize_pdw(vector< vector<float> > pdws, vector< vector<float> >* pdws_normalized, float scale[], int pdw_dims)
{
	for(int i=0; i<(int)pdws.size(); i++)
	{
		vector<float> tmp_vec;
		for(int j=0; j<pdw_dims; j++)
		{
			float tmp = pdws[i][j] / scale[j];
			tmp_vec.push_back(tmp);
		}
		(*pdws_normalized).push_back(tmp_vec);
	}
}

void ART_Complement_Code(vector< vector<float> > pdws_normalized, vector< vector<float> >* ccpdws)
{
	for(int i=0; i<(int)pdws_normalized.size(); i++)
	{
		vector<float> tmp_vec;
		for(int j=1; j<5; j++)
		{
			tmp_vec.push_back(pdws_normalized[i][j]);
			tmp_vec.push_back(1-pdws_normalized[i][j]);
		}
		(*ccpdws).push_back(tmp_vec);
	}
}

vector<float> ART_Activate_Categories(vector<float> input, vector< vector<float> > weight, float bias)
{
	int numCategories = weight.size();
	int numFeatures = weight[0].size();

	vector<float> categoryActivation(numCategories);
	vector<float> matchVector(numFeatures);

	for(int j=0; j<numCategories; j++)
	{
		for(int f=0; f<numFeatures; f++)
		{
			matchVector[f] = min(input[f], weight[j][f]);
		}

		float weightLength = 0.0;
		for(int f=0; f<numFeatures; f++)
		{
			weightLength += weight[j][f];
		}

		float matchVectorLength = 0.0;
		for(int f=0; f<numFeatures; f++)
		{
			matchVectorLength += matchVector[f];
		}

		categoryActivation[j] = matchVectorLength / (bias + weightLength);
	}

	return categoryActivation;

}

vector<size_t> sort_indexes(const vector<float> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

vector< vector<float> > ART_Add_New_Category(vector< vector<float> > weight)
{
	vector< vector<float> > resizedWeight = weight;

	int numCategories = weight.size();
	int numFeatures = weight[0].size();

	vector<float> newCategory(numFeatures,1.0);

	resizedWeight.push_back(newCategory);

	return resizedWeight;
}

int ART_Update_Weights(vector<float> input, vector< vector<float> > &weight, int categoryNumber, float learningRate)
{
	int weightChange = 0;
	int numFeatures = weight[0].size();

	for(int i=0; i<numFeatures; i++)
	{
		if(input[i]<weight[categoryNumber][i])
		{
			weight[categoryNumber][i] = (learningRate * input[i]) + ((1-learningRate)*weight[categoryNumber][i]);
			weightChange = 1;
		}
	}

	return weightChange;
}

float ART_Calculate_Match(vector<float> input, vector<float> weightVector)
{
	float match = 0;

	int numFeatures = input.size();

	vector<float> matchVector(numFeatures);
	float inputLength = 0.0;
	float matchVectorLength = 0.0;

	for(int i=0; i<numFeatures; i++)
	{
		matchVector[i] = min(input[i], weightVector[i]);
		inputLength += input[i];
		matchVectorLength += matchVector[i];
	}

	if(inputLength == 0)
	{
		match = 0.0;
	}
	else
	{
		match = matchVectorLength / inputLength;
	}

	return match;
}

art_network ART_Learn(art_network net, vector< vector<float> > ccpdws, vector<int> &categorization)
{
	art_network new_art_network;

	int numSamples = ccpdws.size();
	int numFeatures = ccpdws[0].size();

	int epochNumber=0;

	for(epochNumber=0; epochNumber<net.numEpochs; epochNumber++)
	{
		int numChanges=0;

		for(int sampleNumber=0; sampleNumber<numSamples; sampleNumber++)
		{
			int weightChange = 0;

			vector<float> currentData = ccpdws[sampleNumber];

			vector<float> categoryActivation = ART_Activate_Categories(currentData, net.weight, net.bias);

			vector<size_t> sortedCategories = sort_indexes(categoryActivation);

			int resonance = 0;
			float match = 0;
			int numSortedCategories = sortedCategories.size();
			int currentSortedIndex = 0;

			while(!resonance)
			{
				if(numSortedCategories == 0)
				{
					vector< vector<float> > resizedWeight = ART_Add_New_Category(net.weight);
					weightChange = ART_Update_Weights(currentData, resizedWeight, 0, net.learningRate);

					net.weight = resizedWeight;
					net.numCategories += 1;
					categorization[sampleNumber] = 0;
					numChanges += 1;
					resonance = 1;
					break;
				}

				int currentCategory = sortedCategories[currentSortedIndex];

				vector<float> currentWeightVector = net.weight[currentCategory];

				match = ART_Calculate_Match(currentData, currentWeightVector);

				if(match>net.vigilance || (match >= 1))
				{
					weightChange = ART_Update_Weights(currentData, net.weight, currentCategory, net.learningRate);

					categorization[sampleNumber] = currentCategory;

					if(weightChange == 1)
					{
						numChanges += 1;
					}

					resonance = 1;
				}
				else
				{
					if(currentSortedIndex == numSortedCategories)
					{
						if(currentSortedIndex == net.maxNumCategories)
						{
							categorization[sampleNumber] = -1;
							resonance = 1;
						}
						else
						{
							vector< vector<float> > resizedWeight = ART_Add_New_Category(net.weight);
							weightChange = ART_Update_Weights(currentData, resizedWeight, currentSortedIndex + 1, net.learningRate);
							net.weight = resizedWeight;
							net.numCategories += 1;
							categorization[sampleNumber] = currentSortedIndex + 1;
							numChanges += 1;
							resonance = 1;
						}
					}
					else
					{
						currentSortedIndex += 1;
					}
				}
			}
		}

		if(numChanges == 0)
		{
			break;
		}
	}

	cout<<"The number of epochs needed was " << epochNumber << endl;

	new_art_network = net;

	return new_art_network;
}

vector<int> ART_Categorize(art_network net, vector< vector<float> > data)
{
	int numSamples = data.size();
	int numFeatures = data[0].size();

	vector<int> categorization(numSamples,-2);

	for(int sampleNumber=0; sampleNumber<numSamples; sampleNumber++)
	{
		vector<float> currentData = data[sampleNumber];

		vector<float> categoryActivation = ART_Activate_Categories(currentData, net.weight, net.bias);

		vector<size_t> sortedCategories = sort_indexes(categoryActivation);

		int resonance = 0;
		float match = 0;
		int numSortedCategories = sortedCategories.size();
		int currentSortedIndex = 0;

		while(!resonance)
		{
			int currentCategory = sortedCategories[currentSortedIndex];

			vector<float> currentWeightVector = net.weight[currentCategory];

			match = ART_Calculate_Match(currentData, currentWeightVector);

			if(match>net.vigilance || (match >= 1))
			{

				categorization[sampleNumber] = currentCategory;
				resonance = 1;
			}
			else
			{
				if(currentSortedIndex == numSortedCategories)
				{
					categorization[sampleNumber] = -1;
					resonance = 1;
				}
				else
				{
					currentSortedIndex = currentSortedIndex + 1;
				}
			}
		}
	}

	return categorization;
}

int main()
{
    struct timeval start, end;
    long learning_time, categorizing_time, seconds, useconds;

	vector< vector<float> > pdws;

    ifstream pdw_file("pdw.csv");

    art_network net = ART_Create_Network();

    cout << "Start importing PDW.\n";

    while(pdw_file)
    {
    	string line;
    	if(!getline(pdw_file, line)) break;

    	istringstream ss(line);

    	vector<float> pdw_record;

    	while(ss)
    	{
    		string cell;
      		if (!getline( ss, cell, ',' )) break;
      		float tmp = atof(cell.c_str());
      		pdw_record.push_back( tmp );
    	}

    	pdws.push_back( pdw_record );

    }

    cout << "Imported " << pdws.size() << " PDWs.\n";

    cout << "Normalizing PDWs.\n";

    vector< vector<float> > pdws_normalized;
    float scale [6] = {1.0, 2000.0, 100.0, 700.0, 360.0, 1.0};
    normalize_pdw(pdws, &pdws_normalized, scale, 6);

    cout << "Normalizing PDWs finished.\n";

    cout << "Generating the Complement Coded PDW.\n";

    vector< vector<float> > ccpdws;
    ART_Complement_Code(pdws_normalized, &ccpdws);

    cout << "Completed generating CCPDWS.\n";

//    cout << "First ccpdws : ";
//    for(int i=0; i<ccpdws[0].size(); i++)
//    {
//    	cout << ccpdws[0][i] << ",";
//    }
//    cout << endl;

    vector<int> categorization(ccpdws.size(),-2);

    // Generate a subvector from the pdws vector for learning process
    vector< vector<float> >::const_iterator first = ccpdws.begin();
    vector< vector<float> >::const_iterator last = ccpdws.begin() + LEARN_SAMPLES;
    vector< vector<float> > ccpdws_learn(first, last);
    vector< vector<float> >::const_iterator last_test = ccpdws.end();
    vector< vector<float> > ccpdws_test(last, last_test);

    cout << "before ART_Learn\n";

    gettimeofday(&start, NULL);

    net = ART_Learn(net, ccpdws_learn, categorization);

    gettimeofday(&end, NULL);

    seconds = end.tv_sec - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec ;
    learning_time = ((seconds)*1000 + useconds/1000.0) + 0.5;

    cout << "Learning time: " << learning_time << endl;

    cout << "after ART_Lern\n";

    cout << "Starting categorization.\n";
    vector<int> newCat(ccpdws.size()-LEARN_SAMPLES,-2);

    gettimeofday(&start, NULL);
    newCat = ART_Categorize(net, ccpdws_test);
    gettimeofday(&end, NULL);

    seconds = end.tv_sec - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec ;
    categorizing_time = ((seconds)*1000 + useconds/1000.0) + 0.5;

    cout << "Categorization time: " << categorizing_time << endl;

    cout << "Finished categorization.\n";


}

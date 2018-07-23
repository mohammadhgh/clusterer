/*
 * pdw_generator.cpp
 *
 *  Created on: Apr 5, 2018
 *      Author: root
 */
#include <vector>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <random>
#include <chrono>

#define END_TIME 10000

using namespace std;

class Emitter
{
private:
	float Freq = 0.0;
	float PRI = 0.0;
	float Amp = 0.0;
	float PW = 0.0;
	float DOA = 0.0;
	float Tag = 0;
	float PRI_Jitter = 0.0;

	float prev_toa = 0.0;

public:
	Emitter(float Freq = 0.0, float PRI = 0.0, float Amp = 0.0, float PW = 0.0, float DOA = 0.0, float Tag = 0.0, float PRI_Jitter = 0.0);
	vector<float> run(float time);

};

Emitter::Emitter(float PRI_ , float Freq_ , float Amp_ , float PW_ , float DOA_ , float Tag_ , float PRI_Jitter_ )
{
	Freq = Freq_;
	PRI = PRI_;
	Amp = Amp_;
	PW = PW_;
	DOA = DOA_;
	Tag = Tag_;
	PRI_Jitter = PRI_Jitter_;

	prev_toa = 0.0;
}

vector<float> Emitter::run(float time_val)
{
	vector<float> pdw;
	float tmp=0;

	srand(time(NULL));

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator (seed);
	std::normal_distribution<double> distribution (0.0,1.0);

	if(time_val-prev_toa>=PRI)
	{
		prev_toa = time_val;

//		float random = (float)rand()/(float) RAND_MAX;
		float random = distribution(generator);

		tmp = time_val + PRI*PRI_Jitter*random;
		pdw.push_back(tmp);

		random = distribution(generator);
		tmp = Freq + random*7;
		pdw.push_back(tmp);

		random = distribution(generator);
		tmp = Amp + random*2;
		pdw.push_back(tmp);

		random = distribution(generator);
		tmp = PW + random*0.2;
		pdw.push_back(tmp);

		random = distribution(generator);
		tmp = DOA + random*5;
		pdw.push_back(tmp);

		pdw.push_back(Tag);
	}

	return pdw;
}

void generate_pdws()
{
	vector< vector<float> > PDWs;

	ofstream pdws_file;
	pdws_file.open("pdws.csv");

	Emitter emitter1(1,	  1000, 50, 10,  15,  1, 0.01);
	Emitter emitter2(1.4,  1200, 60, 1,   198, 2, 0.01);
	Emitter emitter3(2.25, 1400, 90, 100, 355, 3, 0.01);
	Emitter emitter4(4.5,  1600, 30, 5,   49,  4, 0.1);
	Emitter emitter5(13,   1800, 70, 50,  244, 5, 0.1);

	vector<Emitter> emitters = {emitter1,emitter2,emitter3,emitter4,emitter5};

	for(int time_interval=0; time_interval<END_TIME; time_interval++)
	{
		float real_time = time_interval * 1.0;
		for(int emitter_id=0; emitter_id<(int)emitters.size(); emitter_id++)
		{
			vector<float> pdw = emitters[emitter_id].run(real_time);
			if(!pdw.empty())
			{
				for(int i=0; i<(int)pdw.size()-1; i++)
				{
					pdws_file << pdw[i] << ",";
				}
				pdws_file << pdw.back();
				pdws_file << endl;
				PDWs.push_back(pdw);
			}
		}
	}

	pdws_file.close();
}

int main()
{
	cout << "Starting generating PDWs: ..." << endl;

	generate_pdws();

	cout << "Finished generating PDWs file." << endl;

	return 0;
}

#pragma once
#include <functional>
#include <list>
#include <vector>

namespace nw {
	class layer
	{
	private:
		double* weights_;
	
	public:
		int size;
		double* neurons;
		double* biases;		
		
		layer(int size, int nextSize)
			: size(size)
		{
			neurons = new double[size];
			biases = new double[size];
			weights_ = new double[size * nextSize];
		}

		double get_weight(const int x, const int y) const
		{
			return weights_[x * size + y];
		}

		void set_weight(const int x, const int y, const double value) const
		{
			weights_[x * size + y] = value;
		}

		~layer()
		{
			delete[] neurons;
			delete[] biases;
			delete[] weights_;
		}
	};
	
	typedef double (*unary_operator)(double);
	
	class neural_network
	{		
	private:
		double learning_ratio_;
		std::vector<layer*> layers_;
		unary_operator activation_;
		unary_operator derivative_;

		static double get_random()
		{
			return  (static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0;
		}

	public:
		neural_network(double learning_ratio,
			const unary_operator activation,
			const unary_operator derivative,
			const std::vector<int>& sizes)
			: learning_ratio_(learning_ratio), activation_(activation), derivative_(derivative)
		{
			for (size_t i = 0; i < sizes.size(); i++)
			{
				int nextSize = 0;
				if (i < sizes.size() - 1)
					nextSize = sizes[i + 1];

				auto size = sizes[i];
				auto* l = new layer(size, nextSize);

				for(auto j = 0; j<size; j++)
				{
					l->biases[j] = get_random();
					for (auto k = 0; k < nextSize; k++)
					{
						auto w = get_random();
						l->set_weight(j, k, w);
					}
				}							
			}
		}

		std::vector<double> feed_forward(const std::initializer_list<double>& inputs)
		{
			auto* inputLayer = layers_[0];
			std::copy(inputs.begin(), inputs.end(), inputLayer->neurons);

			for (int i = 1; i<layers_.size(); i++)
			{
				auto* cl = layers_[i - 1];
				auto* nl = layers_[i];

				for (int j = 0; j < nl->size; j++)
				{
					nl->neurons[j] = 0.0;

					for (int k = 0; k < cl->size; k++)
						nl->neurons[j] += cl->neurons[k] * cl->get_weight(k, j);

					nl->neurons[j] += nl->biases[j];
					nl->neurons[j] = activation_(nl->neurons[j]);
				}				
			}

			auto* out_layer = layers_[layers_.size() - 1];
			
			return { out_layer->neurons, out_layer->neurons + out_layer->size };
		}

		~neural_network()
		{
			for (auto* l : layers_)
				delete l;
		}
	};
}

#pragma once
#include <functional>
#include <istream>
#include <list>
#include <ostream>
#include <vector>

namespace nn {
	
	template<typename T>
	auto read(std::istream& is, T* value)
	{
		const auto pos = is.tellg();
		is.read(reinterpret_cast<char*>(value), sizeof(*value));
		return static_cast<std::size_t>(is.tellg() - pos);
	}

	template<typename T>
	auto read(std::istream& is, T*& data, size_t* count)
	{
		std::uint32_t size;
		auto pos = read(is, &size);
		*count = size;

		data = new T[size];

		for (auto i = 0U; i < size; i++)
		{
			T value = {};
			pos += read(is, &value);
			data[i] = value;
		}
		
		return pos;
	}

	template<typename T>
	auto write(std::ostream& os, T value) -> std::size_t
	{
		const size_t pos = os.tellp();
		os.write(reinterpret_cast<const char*>(&value), sizeof(value));
		return static_cast<std::size_t>(os.tellp()) - pos;
	}

	template<typename T>
	auto write(std::ostream& os, const T* const data, std::size_t size) -> std::size_t
	{
		const auto pos = os.tellp();
		const auto len = static_cast<std::uint32_t>(size);
		os.write(reinterpret_cast<const char*>(&len), sizeof(len));

		for (auto i = 0U; i < size; i++)
			write(os, data[i]);
			
		return static_cast<std::size_t>(os.tellp() - pos);
	}
	
	class layer
	{
	private:
		double* weights_;
		const int next_size_;

		layer(const int size,
			const int next_size,
			double* weights,
			double* neurons,
			double* biases)
			: weights_(weights), next_size_(next_size), size(size), neurons(neurons), biases(biases)
		{

		}

	public:
		int size;
		double* neurons;
		double* biases;		
		
		layer(const int size, const int next_size)
			: next_size_(next_size), size(size)
		{
			neurons = new double[size];
			biases = new double[size];
			weights_ = new double[size * next_size];
		}

		double get_weight(const int x, const int y) const
		{
			return weights_[x * next_size_ + y];
		}

		void set_weight(const int x, const int y, const double value) const
		{
			weights_[x * next_size_ + y] = value;
		}

		~layer()
		{
			delete[] neurons;
			delete[] biases;
			delete[] weights_;
		}

		auto save(std::ostream& os) const
		{
			auto len = write(os, size);
			len += write(os, next_size_);

			len += write(os, biases, size);
			len += write(os, neurons, size);
			len += write(os, weights_, size * next_size_);
			
			return len;			
		}

		static layer* load(std::istream& is)
		{
			int size = 0;
			read(is, &size);

			int next_size = 0;
			read(is, &next_size);

			double* biases;
			size_t biases_count;
			read(is, biases, &biases_count);

			double* neurons;
			size_t neurons_count;
			read(is, neurons, &neurons_count);

			double* weights;
			size_t weights_count;
			read(is, weights, &weights_count);

			return new layer(size, next_size, weights, neurons, biases);
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

		neural_network(const double learning_ratio,
			const unary_operator activation,
			const unary_operator derivative,
			const std::vector<layer*>& layers)
			: learning_ratio_(learning_ratio), activation_(activation), derivative_(derivative)
		{
			layers_ = { layers };
		}

	public:
		neural_network(const double learning_ratio,
		               const unary_operator activation,
		               const unary_operator derivative,
		               const std::vector<int>& sizes)
			: learning_ratio_(learning_ratio), activation_(activation), derivative_(derivative)
		{
			for (size_t i = 0; i < sizes.size(); i++)
			{
				int next_size = 0;
				if (i < sizes.size() - 1)
					next_size = sizes[i + 1];

				const auto size = sizes[i];
				auto* l = new layer(size, next_size);

				for(auto j = 0; j<size; j++)
				{
					l->biases[j] = get_random();
					for (auto k = 0; k < next_size; k++)
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

			for (auto i = 1; i < static_cast<int>(layers_.size()); i++)
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

		void back_propagation(const std::vector<double> &targets)
		{
			auto* const out_layer = layers_[layers_.size() - 1];
			auto* errors = new double[out_layer->size];

			for (int i = 0; i < out_layer->size; i++)
				errors[i] = targets[i] - out_layer->neurons[i];

			for (int k = static_cast<int>(layers_.size()) - 2; k >= 0; k--)
			{
				auto* cl = layers_[k];
				auto* nl = layers_[k + 1];

				auto* errors_next = new double[cl->size];
				auto* gradient = new double[nl->size];

				for (int i = 0; i < nl->size; i++)				
					gradient[i] = errors[i] * derivative_(nl->neurons[i]) * learning_ratio_;

				auto* deltas = new double[nl->size * cl->size];
				for (auto i = 0; i < nl->size; i++)
					for (auto j = 0; i < cl->size; j++)
						deltas[i * cl->size + j] = gradient[i] * cl->neurons[j];

				for(int i = 0; i<cl->size; i++)
				{
					errors_next[i] = 0;
					for (int j = 0; j < nl->size; j++)
						errors_next[i] += cl->get_weight(i, j) * errors[j];
				}

				delete[] errors;
				errors = new double[cl->size];

				std::copy_n(errors_next, cl->size, errors);

				auto* const weights_new = new double[cl->size * nl->size];

				for (int i = 0; i < nl->size; i++)
					for (int j = 0; j < cl->size; j++)
						weights_new[j * nl->size + i] = cl->get_weight(j, i) + deltas[i * cl->size + j];

				for (int i = 0; i < cl->size; i++)
					for (int j = 0; j < nl->size; j++)
						cl->set_weight(i, j, weights_new[i * nl->size + j]);

				for (int i = 0; i < nl->size; i++)
					nl->biases[i] += gradient[i];

				delete[] weights_new;
				delete[] deltas;
				delete[] gradient;
				delete[] errors_next;
			}

			delete[] errors;
		}

		~neural_network()
		{
			for (auto* l : layers_)
				delete l;
		}

		auto save(std::ostream& os)
		{
			write(os, learning_ratio_);
			
			const auto pos = os.tellp();

			const auto len = static_cast<std::uint32_t>(layers_.size());
			os.write(reinterpret_cast<const char*>(&len), sizeof(len));

			for (auto* l : layers_)
				l->save(os);
			
			return static_cast<std::size_t>(os.tellp() - pos);
		}
		
		static neural_network* load(std::istream& is, 
			const unary_operator activation,
			const unary_operator derivative)
		{
			auto learning_ratio = 0.0;
			read(is, &learning_ratio);

			std::uint32_t layers_count = 0;
			read(is, &layers_count);

			std::vector<layer*> layers;

			for (auto i = 0U; i < layers_count; i++)
				layers.push_back(layer::load(is));
			
			return new neural_network(learning_ratio, activation, derivative, layers);
		}
	};
}

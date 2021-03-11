#include <fstream>
#include <map>
#include <filesystem>
#include <algorithm>
#include <execution>

#include <SDL_image.h>

#include "../include/main.hpp"

void on_loop(long current_tick)
{
	
}

void on_render()
{
	SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0xFF);

	SDL_RenderClear(renderer);

	SDL_SetRenderTarget(renderer, texture);

	//TODO: Render to texture
	
	if (is_clear)
		SDL_RenderClear(renderer);

	is_clear = false;
	SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
	
	if (is_move)
		SDL_RenderDrawLine(renderer,
			pt_picture_start.x, pt_picture_start.y,
			pt_picture_end.x, pt_picture_end.y);

	SDL_SetRenderTarget(renderer, nullptr);

	SDL_Rect rect = { 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT };

	SDL_RenderCopy(renderer, texture, nullptr, &rect);

	SDL_RenderPresent(renderer);	
}

bool on_event(SDL_Event event)
{
	switch (event.type)
	{
	case SDL_KEYUP:
		switch (event.key.keysym.sym)
		{
		case SDLK_ESCAPE:
			return true;
		case SDLK_SPACE:
			is_clear = true;
			break;
		}
		break;
	case SDL_MOUSEBUTTONDOWN:
		is_pressed = true;
		break;

	case SDL_MOUSEBUTTONUP:
		is_pressed = false;
		is_move = false;
		break;
	case SDL_MOUSEMOTION:
		if (!is_pressed)
			return false;		
			
		auto dx = static_cast<double>(PICTURE_WIDTH) / static_cast<double>(WINDOW_WIDTH);
		auto dy = static_cast<double>(PICTURE_HEIGHT) / static_cast<double>(WINDOW_HEIGHT);

		pt_picture_end = { static_cast<int>(dx * event.motion.x), static_cast<int>(dy * event.motion.y) };

		if (!is_move)
			pt_picture_start = pt_picture_end;

		is_move = true;	
		break;		
	}

	return false;
}

bool init()
{
	if (!IMG_Init(IMG_INIT_PNG))
		return false;
	
	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS))
		return false;

	window = SDL_CreateWindow("Digits recognizer",
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		WINDOW_WIDTH,
		WINDOW_HEIGHT,
		SDL_WINDOW_SHOWN);

	if (window == nullptr)
		return false;

	renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

	if (renderer == nullptr)
		return false;

	texture = SDL_CreateTexture(renderer,
		SDL_PIXELFORMAT_RGBA8888,
		SDL_TEXTUREACCESS_TARGET,
		PICTURE_WIDTH,
		PICTURE_HEIGHT);
	
	return true;
}

double sigmoid(double x)
{
	return 1 / (1 + std::exp(-x));
}

double dsigmoid(double y)
{
	return y * (1 - y);
}

double* load_image(const char* file_name, size_t* size)
{
	auto* image = IMG_Load(file_name);

	auto* pixels = static_cast<std::uint8_t*>(image->pixels);
	*size = image->h * image->w;
	
	auto* values = new double[*size];

	for (auto i = 0U; i < *size; i++)
	{
		auto p = pixels[i];
		auto v = (p & 0xFF) / 255.0;
		values[i] = v;
	}
	
	SDL_FreeSurface(image);
	
	return values;
}

std::vector<digit_sample>* load_samples(const char* path)
{
	std::filesystem::directory_iterator train_directory(path);

	std::vector<std::filesystem::path> train_files;

	for (const auto& entry : train_directory)
	{
		if (!entry.is_regular_file())
			continue;

		train_files.push_back(std::filesystem::absolute(entry.path()));
	}

	auto* samples = new std::vector<digit_sample>();

	std::mutex m;

	std::for_each(std::execution::par_unseq,
		train_files.begin(),
		train_files.end(),
		[&samples, &m](const auto& p)
		{
			const std::string num = { p.filename().string()[10] };
			size_t size;
			auto* image = load_image(p.string().c_str(), &size);

			m.lock();
			samples->push_back({ std::atoi(num.c_str()), size, image });
			m.unlock();
		});

	return samples;
}

void learn_digits_nn(nn::neural_network* nn, int epochs)
{
	auto* samples = load_samples("../../../../dataset/train");
	
	for (auto i = 0u; i < epochs; i++)
	{
		int right = 0;
		double error_sum = 0;
		int batch_size = 100;

		for (auto j = 0; j < batch_size; j++)
		{
			const auto& s = samples->at(rand() % samples->size());
			const auto targets_size = 10;
			double targets[targets_size] = { 0.0 };
			
			targets[s.digit] = 1.0;

			std::vector<double> inputs(s.data, s.data + s.size);
			
			auto outputs = nn->feed_forward(inputs);
			int max_digit = 0;
			double max_digit_weight = -1;

			for (int k = 0; k < 10; k++) {
				if (outputs[k] > max_digit_weight) {
					max_digit_weight = outputs[k];
					max_digit = k;
				}
			}

			if (s.digit == max_digit) 
				right++;
			
			for (int k = 0; k < 10; k++) {
				error_sum += (targets[k] - outputs[k]) * (targets[k] - outputs[k]);
			}

			const std::initializer_list<double> targets_list(targets, targets + targets_size);
			nn->back_propagation(targets_list);
		}
	}

	std::for_each(std::execution::par_unseq,
		samples->begin(),
		samples->end(),
		[](const auto& s)
		{
			delete[] s.data;
		});

	delete samples;
}

nn::neural_network* create_digits_nn()
{
	nn::neural_network* nn = nullptr;	
	const char* model_file = "digits_nn_model.dat";

	if (std::filesystem::exists(model_file))
	{
		std::ifstream is(model_file, std::ios::binary);
		nn = nn::neural_network::load(is, &sigmoid, &dsigmoid);
		is.close();
	}
	else 
	{
		nn = new nn::neural_network(0.001,
			&sigmoid,
			&dsigmoid,
			{ 784, 512, 128, 32, 10 });
		
		learn_digits_nn(nn, 1000);

		std::ofstream os(model_file, std::ios::binary);
		nn->save(os);
		os.close();
	}
		
	return nn;
}

int main(int argc, char* argv[])
{
	if (!init())
		return 1;

	auto learn_task = std::async(std::launch::async, create_digits_nn);
	
	auto quit = false;
	long current_tick = 0;
	nn::neural_network* nn = nullptr;
	while(!quit)
	{
		pt_picture_start = pt_picture_end;
		SDL_Event event;
		while(SDL_PollEvent(&event))
		{
			switch (event.type)
			{
			case SDL_QUIT:
				quit = true;
				break;
			default:
				if (on_event(event))
					quit = true;
				break;
			}
			
			SDL_PumpEvents();
		}

		current_tick = SDL_GetTicks();

		if (nn == nullptr && learn_task._Is_ready())
			nn = learn_task.get();
		
		on_loop(current_tick);
		on_render();
	}

	delete nn;
	
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	IMG_Quit();
	
	return 0;
}

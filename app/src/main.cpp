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

double* load_image(const char* file_name)
{
	auto* image = IMG_Load(file_name);

	auto* pixels = static_cast<char*>(image->pixels);
	const size_t size = image->h * image->w;
	
	auto* values = new double[size];

	for (auto i = 0U; i < size; i++)
	{
		auto p = pixels[i];
		auto v = (p & 0xFF) / 255.0;
		values[i] = v;
	}
	SDL_FreeSurface(image);
	
	return values;
}

nn::neural_network* create_digits_nn()
{
	std::filesystem::directory_iterator train_directory("../../../../dataset/train");

	std::vector<std::filesystem::path> train_files;

	for(const auto& entry :train_directory)
	{
		if(!entry.is_regular_file())
			continue;

		train_files.push_back(std::filesystem::absolute(entry.path()));
	}
	
	std::vector<digit_sample> samples;
	
	std::for_each(train_files.begin(),
		train_files.end(), [&samples](const auto& p)
		{
			std::string std_num = { p.filename().string()[10] };

			auto* image = load_image(p.string().c_str());

			samples.push_back({ atoi(std_num.c_str()), image });
		});	
	
	auto* nn = new nn::neural_network(0.001,
		&sigmoid,
		&dsigmoid,
		{ 784, 512, 128, 32, 10 });

	const char* model_file = "digits_nn_model.dat";

	std::ofstream os(model_file, std::ios::binary);
	nn->save(os);
	os.close();

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

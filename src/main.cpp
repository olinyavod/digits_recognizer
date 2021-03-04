#include <SDL.h>

#include "../include/neural_network.hpp"

const int WINDOW_WIDTH = 640;
const int WINDOW_HEIGHT = 480;
const int PICTURE_WIDTH = 28;
const int PICTURE_HEIGHT = 28;

SDL_Renderer* renderer = nullptr;
SDL_Window* window = nullptr;
SDL_Texture* texture = nullptr;
SDL_Point pt_picture_start, pt_picture_end;
bool is_move = false , is_clear = false, is_pressed = false;

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

int main(int argc, char* argv[])
{
	if (!init())
		return 1;

	auto quit = false;
	long current_tick = 0;
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

		on_loop(current_tick);
		on_render();
	}
	
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}

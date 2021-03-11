#pragma once

#include <future>

#include <SDL.h>

#include "neural_network.hpp"

struct digit_sample
{
	int digit;
	size_t size;
	double* data;
};

const int WINDOW_WIDTH = 640;
const int WINDOW_HEIGHT = 480;
const int PICTURE_WIDTH = 28;
const int PICTURE_HEIGHT = 28;

SDL_Renderer* renderer = nullptr;
SDL_Window* window = nullptr;
SDL_Texture* texture = nullptr;
SDL_Point pt_picture_start, pt_picture_end;
bool is_move = false, is_clear = false, is_pressed = false;


void on_loop(long current_tick);

void on_render();

bool on_event(SDL_Event event);
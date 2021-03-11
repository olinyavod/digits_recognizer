#pragma once

#include <future>

#include "neural_network.hpp"

enum app_state
{
	start,
	loading,
	learning,
	paint
};

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


void on_loop(long current_tick);

void on_render();

bool on_event(SDL_Event event);

double* load_inputs_from_surface(SDL_Surface* image, size_t* size);
#include <SFML/Graphics.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <chrono>

#define N 256
#define SPHERE_COUNT 16
#define T 60
#define RAD 10
#define A 32
const float PI = acos(-1);

struct sphere {
	int R;
	int start;
	int r, g, b;
};

struct color {
	int r, g, b;
};

void generate(sphere* a) {
	int dx = (T + SPHERE_COUNT - 1) / SPHERE_COUNT;
	for (int i = 0; i < SPHERE_COUNT; ++i) {
		a[i].R = RAD;
		if (i == 0) {
			a[i].start = 0;
		}
		else {
			a[i].start = std::min(a[i - 1].start + dx, T);
		}

		a[i].r = rand() % 256;
		a[i].g = rand() % 256;
		a[i].b = rand() % 256;
	}
}

void calculate(float* gpuCardioParams) {
	float* tmp;
	tmp = (float*)malloc(T * sizeof(float));
	tmp[0] = 0;
	tmp[T - 1] = 0;

	float dx = (2. * PI) / (float)(T);
	for (int i = 1; i < T - 1; ++i) {
		tmp[i] = tmp[i - 1] + dx;
	}

	cudaMemcpy(gpuCardioParams, tmp, T * sizeof(float), cudaMemcpyHostToDevice);
}

void cpuResetColors(color* a, sf::Image& pic) {
	for (int y = 0; y < pic.getSize().y; ++y) {
		for (int x = 0; x < pic.getSize().x; ++x) {
			int it = x * N + y;
			sf::Color u(a[it].r, a[it].g, a[it].b);
			pic.setPixel(x, y, u);
		}
	}
}


__global__ void gpuResetColors(color* a, sphere* s, float* cardio, int dt) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int it = x * N + y;

	a[it].r = 0;
	a[it].g = 0;
	a[it].b = 0;

	int sphIt = -1, dist = 0, x0, y0;
	for (int i = 0; i < SPHERE_COUNT; ++i) {
		int u = (s[i].start + dt) % T;
		float alph = cardio[u];

		x0 = 2. * A * cosf(alph) - A * cosf(2. * alph) + N/2;
		y0 = 2. * A * sinf(alph) - A * sinf(2. * alph) + N/2;

		dist = (x - x0) * (x - x0) + (y - y0) * (y - y0);
		if (dist > RAD * RAD) {
			continue;
		}

		sphIt = i;
		break;
	}

	if (sphIt != -1) {
		float d = (float)dist / (float)(s[sphIt].R * s[sphIt].R);
		d = 1. - d;
		a[it].r = s[sphIt].r * d;
		a[it].g = s[sphIt].g * d;
		a[it].b = s[sphIt].b * d;
	}
	
}


int main() {
	srand(time(NULL));

	sphere* cpuSpheres;
	sphere* gpuSpheres;

	color* cpuColors;
	color* gpuColors;

	float* gpuCardioParams;
	cudaMalloc(&gpuCardioParams, T * sizeof(float));
	calculate(gpuCardioParams);

	cpuSpheres = (sphere*)malloc(SPHERE_COUNT * sizeof(sphere));
	cudaMalloc(&gpuSpheres, SPHERE_COUNT * sizeof(sphere));

	cpuColors = (color*)malloc(N * N * sizeof(color));
	cudaMalloc(&gpuColors, N * N * sizeof(color));

	generate(cpuSpheres);
	cudaMemcpy(gpuSpheres, cpuSpheres, SPHERE_COUNT * sizeof(sphere), cudaMemcpyHostToDevice);

	sf::Image pic;
	pic.create(N, N);

	dim3 blocks(N, N);
	sf::RenderWindow window(sf::VideoMode(N, N), "spheres");

	int cnt = 0, cntType = 0;
	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				window.close();
			}
			
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
				generate(cpuSpheres);
				cudaMemcpy(gpuSpheres, cpuSpheres, SPHERE_COUNT * sizeof(sphere), cudaMemcpyHostToDevice);
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {
				cntType = 1;
			}
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
				cntType = 2;
			}
		}

		//
		gpuResetColors<<< blocks, 1 >>>(gpuColors, gpuSpheres, gpuCardioParams, cnt);

		cudaMemcpy(cpuColors, gpuColors, N * N * sizeof(color), cudaMemcpyDeviceToHost);
		cpuResetColors(cpuColors, pic);


		_sleep(10);
		
		if (cntType == 1) {
			cnt = (cnt + 1) % T;
		}
		if (cntType == 2) {
			cnt = cnt - 1;
			if (cnt == -1) {
				cnt = T - 1;
			}
		}

		sf::Texture texture;
		texture.create(N, N);
		texture.update(pic);

		sf::Sprite sprite;
		sprite.setTexture(texture);

		window.clear();
		window.draw(sprite);
		window.display();
	}

	free(cpuSpheres);
	cudaFree(gpuSpheres);

	free(cpuColors);
	cudaFree(gpuColors);

	cudaFree(gpuCardioParams);
	return 0;
}


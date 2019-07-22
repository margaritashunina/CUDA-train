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

#define N 512
#define SPHERE_COUNT 20

struct sphere {
	int R;
	int x, y, z;
	int r, g, b;
};

struct color {
	int r, g, b;
};

void generate(sphere* a) {
	for (int i = 0; i < SPHERE_COUNT; ++i) {
		a[i].R = rand() % (N / 4) + 1;
		a[i].x = rand() % N;
		a[i].y = rand() % N;
		a[i].z = rand() % 100;

		a[i].r = rand() % 256;
		a[i].g = rand() % 256;
		a[i].b = rand() % 256;
	}
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


__global__ void gpuResetColors(color* a, sphere* s) {
	int x = blockIdx.x;
	int y = threadIdx.x;

	int it = x * N + y;

	a[it].r = 0;
	a[it].g = 0;
	a[it].b = 0;

	int sphIt = -1, neededDist = 0, mxz = 0;
	for (int i = 0; i < SPHERE_COUNT; ++i) {
		int dist = (s[i].x - x) * (s[i].x - x) + (s[i].y - y) * (s[i].y - y);
		
		if (dist > s[i].R * s[i].R) {
			continue;
		}

		int z = s[i].R * s[i].R - dist;

		if (z > mxz) {
			neededDist = dist;
			mxz = z;
			sphIt = i;
		}
	}

	if (sphIt != -1) {
		float d = (float)neededDist / (float)(s[sphIt].R * s[sphIt].R);
		d = 1. - d;
		a[it].r = (float)s[sphIt].r * d;
		a[it].g = (float)s[sphIt].g * d;
		a[it].b = (float)s[sphIt].b * d;
	}
}

int main() {
	srand(time(NULL));

	sphere* cpuSpheres;
	sphere* gpuSpheres;

	color* cpuColors;
	color* gpuColors;

	cpuSpheres = (sphere*)malloc(SPHERE_COUNT * sizeof(sphere));
	cudaMalloc(&gpuSpheres, SPHERE_COUNT * sizeof(sphere));

	cpuColors = (color*)malloc(N * N * sizeof(color));
	cudaMalloc(&gpuColors, N * N * sizeof(color));

	generate(cpuSpheres);
	cudaMemcpy(gpuSpheres, cpuSpheres, SPHERE_COUNT * sizeof(sphere), cudaMemcpyHostToDevice);

	sf::Image pic;
	pic.create(N, N);


	sf::RenderWindow window(sf::VideoMode(N, N), "spheres");
	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				window.close();
			}
			if (event.type == sf::Event::KeyPressed) {
				generate(cpuSpheres);
				cudaMemcpy(gpuSpheres, cpuSpheres, SPHERE_COUNT * sizeof(sphere), cudaMemcpyHostToDevice);
			}
		}

		//
		gpuResetColors<<< N, N >>>(gpuColors, gpuSpheres);
		cudaMemcpy(cpuColors, gpuColors, N * N * sizeof(color), cudaMemcpyDeviceToHost);
		cpuResetColors(cpuColors, pic);

		_sleep(500);

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

	return 0;
}


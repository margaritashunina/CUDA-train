#include <SFML/Graphics.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <chrono>

#define N 280

const int reds[] = { 255, 255, 255, 0, 0, 0, 255 };
const int greens[] = { 0, 128, 255, 255, 128, 255, 0 };
const int blues[] = { 0, 0, 0, 0, 255, 255, 255 };

void cpuColorReset(sf::Image& pic, int* cpuColorIt) {
	for (int y = 0; y < pic.getSize().y; ++y) {
		for (int x = 0; x < pic.getSize().x; ++x) {
			int it = x * N + y;
			int wich = cpuColorIt[it];
			pic.setPixel(x, y, sf::Color(reds[wich], greens[wich], blues[wich]));
		}
	}
}

__global__ void gpuColorReset(int* gpuColorIt, int dx) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	int it = x * N + y;
	
	gpuColorIt[it] = (y / 7 + dx) % 7;
}

int main() {
	int* cudaColorIt;
	int* cpuColorIt;

	cudaMalloc(&cudaColorIt, N * N * sizeof(int));
	cpuColorIt = (int*)malloc(N * N * sizeof(int));

	sf::Image rainbow;
	rainbow.create(N, N);

	gpuColorReset <<< N, N >>> (cudaColorIt, 0);

	cudaMemcpy(cpuColorIt, cudaColorIt, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	cpuColorReset(rainbow, cpuColorIt);

	int cnt = 0;
	sf::RenderWindow window(sf::VideoMode(N, N), "first try");
	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				window.close();
			}
		}

		//
		gpuColorReset << < N, N >> > (cudaColorIt, cnt);

		cudaMemcpy(cpuColorIt, cudaColorIt, N * N * sizeof(int), cudaMemcpyDeviceToHost);
		cpuColorReset(rainbow, cpuColorIt);
		//

		_sleep(100);

		cnt = (cnt + 1) % 7;

		sf::Texture texture;
		texture.create(N, N);
		texture.update(rainbow);

		sf::Sprite sprite;
		sprite.setTexture(texture);

		window.clear();
		window.draw(sprite);
		window.display();
	}


	cudaFree(cudaColorIt);
	free(cpuColorIt);
	return 0;
}


/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	if (is_initialized) { return; }

	num_particles = 100;

	double std_x = std[0];
  	double std_y = std[1];
  	double std_theta = std[2];

	normal_distribution<double> normal_dist_x(x, std_x);
	normal_distribution<double> normal_dist_y(y, std_y);
	normal_distribution<double> normal_dist_theta(theta, std_theta);

	for (int i = 0; i < num_particles; i++) {
		
		Particle p;
		p.id = i;
		p.x = normal_dist_x(gen);
		p.y = normal_dist_y(gen);
		p.theta = normal_dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	double std_x = std_pos[0];
  	double std_y = std_pos[1];
  	double std_theta = std_pos[2];

	normal_distribution<double> normal_dist_x(0, std_x);
	normal_distribution<double> normal_dist_y(0, std_y);
	normal_distribution<double> normal_dist_theta(0, std_theta);

	for (int i = 0; i < num_particles; i++) {

		if (fabs(yaw_rate) < 0.00001) {  
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		} else {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		particles[i].x += normal_dist_x(gen);
		particles[i].y += normal_dist_x(gen);
		particles[i].theta += normal_dist_x(gen);
  	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.


	for (int i = 0; i < observations.size(); i++) {
		
		LandmarkObs observation = observations[i];
		double min_distance = numeric_limits<double>::max();
		int map_id = -1;

		for (int j = 0; j < predicted.size(); j++) {
			
			LandmarkObs prediction = predicted[i];
			double distance = dist(observation.x, observation.y, prediction.x, prediction.y);

			if (distance < min_distance) {
				min_distance = distance;
				map_id = prediction.id;
			}
		}
		observation.id = map_id;
	
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; i++) {
		
		Particle p = particles[i];
		vector<LandmarkObs> landmarks_in_range;

		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			float landmark_x = map_landmarks.landmark_list[j].x_f;
			float landmark_y = map_landmarks.landmark_list[j].y_f;
			float landmark_id = map_landmarks.landmark_list[j].id_i;

			double dx = fabs(landmark_x - p.x);
			double dy = fabs(landmark_y - p.y);

			if (dx <= sensor_range && dy <= sensor_range) {
				LandmarkObs lm = LandmarkObs{ landmark_id, landmark_x, landmark_y };
				landmarks_in_range.push_back(lm);
			}
		}

		vector<LandmarkObs> transformed_observations;

		for (int j = 0; j < observations.size(); j++) {
			
			double t_x = cos(p.theta) * observations[j].x - sin(p.theta) * observations[j].y + p.x;
			double t_y = sin(p.theta) * observations[j].x + cos(p.theta) * observations[j].y + p.y;
			LandmarkObs tr_ob = LandmarkObs{ observations[j].id, t_x, t_y };

			transformed_observations.push_back(tr_ob);
		}

		dataAssociation(landmarks_in_range, transformed_observations);

		p.weight = 1.0;

		for (int j = 0; j < transformed_observations.size(); j++) {

			LandmarkObs tr_ob = transformed_observations[j];
			LandmarkObs lm;

			for (unsigned int k = 0; k < landmarks_in_range.size(); k++) {
				if (landmarks_in_range[k].id == tr_ob.id) {
					lm.x = landmarks_in_range[k].x;
					lm.y = landmarks_in_range[k].y;
				}
			}

			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			double dx = lm.x - tr_ob.x;
			double dy = lm.y - tr_ob.y;

			double weight = (1 / (2*M_PI*std_x*std_y)) * exp(-(pow(dx,2) / (2 * pow(std_x, 2)) + (pow(dy,2) / (2 * pow(std_y, 2)))));

			particles[i].weight *= weight;
		}

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

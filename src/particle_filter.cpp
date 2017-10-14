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
#include <fstream>

#include "particle_filter.h"

#define NUM_PARTICLES 20
#define EPS 0.0001

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = NUM_PARTICLES;

  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  std::default_random_engine gen;
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  weights.resize(num_particles);
  particles.resize(num_particles);

  for (int i = 0; i < num_particles; ++i) {
    weights[i] = 1.0;

    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
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

  std::default_random_engine gen;
  normal_distribution<double> dist_x(0, std_x);
  normal_distribution<double> dist_y(0, std_y);
  normal_distribution<double> dist_theta(0, std_theta);

  for (int i = 0; i < num_particles; ++i) {
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    if (fabs(yaw_rate) < EPS) {
      double v_dt = velocity * delta_t;

      x += v_dt * cos(theta);
      y += v_dt * sin(theta);
    } 
    else {
      double yr_dt = yaw_rate * delta_t;
      double v_yr = velocity / yaw_rate;

      x += v_yr * (sin(theta + yr_dt) - sin(theta));
      y += v_yr * (cos(theta) - cos(theta + yr_dt));
      theta += yr_dt;
    }

    particles[i].x = x + dist_x(gen);
    particles[i].y = y + dist_y(gen);
    particles[i].theta = theta + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  for (int i = 0; i < observations.size(); ++i) {
    double min_distance = -1;

    for (int j = 0; j < predicted.size(); ++j) {
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      if (distance < min_distance || min_distance < 0) {
        min_distance = distance;
        observations[i].id = predicted[j].id;
      }
    }
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

  double sensor_range_2 = sensor_range * sensor_range;
  double std_x = std_landmark[0];
  double std_y = std_landmark[1];
  double gauss_norm = 1.0 / (2.0 * M_PI * std_x * std_y);

  for (int i = 0; i < particles.size(); ++i) {
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    // get list of all landmarks in sensor range to particle
    vector<LandmarkObs> in_range;
    for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      double lm_x = map_landmarks.landmark_list[j].x_f;
      double lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;

      double distance = dist(p_x, p_y, lm_x, lm_y);
      if (distance <= sensor_range_2) {
        LandmarkObs obs = { lm_id, lm_x, lm_y };
        in_range.push_back(obs);
      }
    }

    // convert all observations from vehicle coords to map coords.
    vector<LandmarkObs> map_observations;
    for (int j = 0; j < observations.size(); ++j) {
      double o_x = observations[j].x;
      double o_y = observations[j].y;
      int o_id = observations[j].id;

      double x_m = p_x + cos(p_theta)*o_x - sin(p_theta)*o_y;
      double y_m = p_y + sin(p_theta)*o_x + cos(p_theta)*o_y;
      LandmarkObs obs = { o_id, x_m, y_m };
      map_observations.push_back(obs);
    }

    // link map_observations to closest in_range landmark for particle
    dataAssociation(in_range, map_observations);

    // debug: set associations.
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    for (int i = 0; i < map_observations.size(); ++i) {
      associations.push_back(map_observations[i].id);
      sense_x.push_back(map_observations[i].x);
      sense_y.push_back(map_observations[i].y);
    }
    particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);

    // update particle weight using the product of the multi-variate Gaussian distribution of observations.
    double p_weight = 1.0;
    for (int j = 0; j < map_observations.size(); ++j) {
      LandmarkObs observation = map_observations[j];

      // find landmark associated with observation
      LandmarkObs landmark;
      for (int k = 0; k < in_range.size(); ++k) {
        if (in_range[k].id == observation.id) {
          landmark = in_range[k];
          break;
        }
      }

      // calculate the observations weight
      double d_x = landmark.x - observation.x;
      double d_y = landmark.y - observation.y;
      double exponent = (d_x*d_x) / (2.0 * std_x * std_x) + (d_y*d_y) / (2.0 * std_y * std_y);
      double weight = gauss_norm * exp(-exponent);

      p_weight *= gauss_norm * exp(-exponent);
    }

    particles[i].weight = p_weight;
    weights[i] = p_weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  vector<Particle> n_particles;
  std::default_random_engine gen;
  std::discrete_distribution<int> dist_w(weights.begin(), weights.end());

  for (int i = 0; i < num_particles; ++i) {
    n_particles.push_back(particles[dist_w(gen)]);
  }
  particles = n_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations,
  std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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

import numpy as np
import scipy.linalg


chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):

    def __init__(self):
        ndim, dt = 4, 1.

        # Matriks Kalman Filtera
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Bobot untuk pergerakan box prediksi
        self._weight_position = 1. / 20
        self._weight_velocity = 1. / 160

    def initiate(self, measurement):
        """
            Buat Track Awal

            Parameters
            ----------
            measurement: array
                Koordinat BBox (x, y, a, h)
                dengan 
                    - x dan y adalah titik tengah
                    - a, Aspect Ratio 
        """
        mean_pos = measurement
        mean_velo = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_velo]

        std = [
            # the center point x
            2 * self._weight_position * measurement[0],

            # the center point y
            2 * self._weight_position * measurement[1],

            # the ratio of width/height
            1 * measurement[2],

            # the height
            2 * self._weight_position * measurement[3],

            10 * self._weight_velocity * measurement[0],
            10 * self._weight_velocity * measurement[1],
            0.1 * measurement[2],
            10 * self._weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
            Prediksi track baru pakai Kalman Filter

            Parameters
            ----------
            mean : array 
                vektor rata-rata dimensi 8x8 dari keadaan box (objek) pada langkah atau keadaan sebelumnya
            covariance : array
                Matriks kovarians dimensi 8x8 dari keadaan objek pada langkah waktu sebelumnya.
        """

        mat_pos = [
            self._weight_position * mean[0],
            self._weight_position * mean[1],
            1 * mean[2],
            self._weight_position * mean[3]
        ]

        mat_velo = [
            self._weight_velocity * mean[0],
            self._weight_velocity * mean[1],
            0.1 * mean[2],
            self._weight_velocity * mean[3]
        ]

        r_ = np.r_[mat_pos, mat_velo]
        square = np.square(r_)
        motion_cov = np.diag(square)

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._motion_mat, covariance, self._motion_mat.T)
        )

        return mean, covariance

    def project(self, mean, covariance):
        """
            Distribusi State ke Measurement Space

            Parameters
            ----------
            mean : array
                State Matriks (8 dimensi array)
            covariance: array
                Matriks Covariance (8x8 dimensi array)
        """
        mat = [
            self._weight_position * mean[0],
            self._weight_position * mean[1],
            0.1 * mean[2],
            self._weight_position * mean[3]
        ]

        innov_square = np.square(mat)
        innovation_cov = np.diag(innov_square)

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )

        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """
            Run Kalman Filter Correction
        """

        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )

        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False
        )
        kalman_gain = kalman_gain.T

        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):

        mean, covariance = self.project(mean, covariance)

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)

        d = measurements - mean

        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True
        )

        squred_maha = np.sum(z * z, axis=0)

        return squred_maha

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class RoomModel:
    _air_z0 = 420  # characteristic impedance of air [PaS/m]
    _air_speed = 343  # speed of sound in air [m/s]

    def __init__(
        self,
        dist_to_observer: float,
        source_intensity: float,
        frequency: float,
        n_space: int = 50000,
        n_time: int = 2000,
        time_to_sim: float = 0.1,
    ):
        """
        RoomModel object. Simulates plane wave propagation in one dimension. Source is at x = 0[m].

        Parameters
        ----------
        dist_to_observer : float
            Distance from source (x = 0[m]) to observer. Is also the boundry of the simulation.
        source_intensity : float
            Source intensity in dB (intensity).
        frequency : float
            Frequency of tone [Hz].
        n_space : int, optional
            Number of spatial samples, by default 50000
        n_time : int, optional
            Number of temporal samples, by default 200
        time_to_sim : float, optional
            Simulation duration [s], by default 0.1
        """
        self.dist_to_observer = dist_to_observer
        self.n_space = n_space
        self.n_time = n_time

        self.i_0 = 10e-12  # reference value for intensity dB scale [W/m^2]
        self.source_intensity = self.i_0 * 10 ** (
            source_intensity / 10
        )  # converted from dB to actual intensity value
        self._omega = 2 * np.pi * frequency

        # Dictionary for keeping track of impedant objects
        self._object_dict = {}
        # Array defining char. impedance and speed of sound in model, filled with values for air by default
        self._model_val_array = np.full(
            (self.n_space, 2), (self._air_z0, self._air_speed)
        )

        # simulation and plot parameters:
        self._t_arr = np.linspace(0, time_to_sim, n_time)
        self._x_arr = np.linspace(0, dist_to_observer, n_space)

    # private methods
    def _calc_coefs(self, Z_1: float, Z_2: float) -> tuple[float, float]:
        R_i = ((Z_1 - Z_2) ** 2) / ((Z_1 + Z_2) ** 2)  # reflection coefficient
        T_i = 1 - R_i  # transmission coefficient
        return R_i, T_i

    # public methods
    def add_impedant_object(
        self, name: str, start: float, end: float, char_impedance: float, density: float
    ) -> None:
        """Add an object with a certain acoustic impedance.

        Parameters
        ----------
        name : str
            Name of object.
        start : float
            Start location of object [m]
        end : float
            end location of object [m]
        char_impedance : float
            Characteristic impedance of object [PaS/m]
        density : float
            Density of object [kg/m^3]
        """
        if name in self._object_dict:
            raise ValueError(f"{name} already an object in model.")

        start_pos_index = int(start / self.dist_to_observer * self.n_space)
        end_pos_index = int(end / self.dist_to_observer * self.n_space)
        sound_speed = char_impedance / density  # speed of sound in given medium [m/s]

        self._object_dict[name] = (start_pos_index, end_pos_index)

        for i in range(start_pos_index, end_pos_index):
            self._model_val_array[i][0] = char_impedance
            self._model_val_array[i][1] = sound_speed

    def remove_impedant_object(self, name: str) -> None:
        """Remove specified object from model.

        Parameters
        ----------
        name : str
            Name of object.

        Raises
        ------
        ValueError
            If object is not in model.
        """

        if name not in self._object_dict:
            raise ValueError(f"{name} is not an object in the model.")

        object_indeces = self._object_dict.pop(name)

        start_pos_index = object_indeces[0]
        end_pos_index = object_indeces[1]

        self._model_val_array[start_pos_index:end_pos_index] = (
            self._air_z0,
            self._air_speed,
        )

    def calc_intensity(self) -> None:
        """Simulate system."""
        wave_anal_sol = np.zeros((self.n_space, self.n_time))
        imp_arr = self._model_val_array
        max_amplitude = self.source_intensity
        sound_speed = imp_arr[0][1]  # first value of speed as default
        wavenumber = self._omega / sound_speed

        for i in range(len(imp_arr) - 1):
            current_imp = imp_arr[i][0]
            next_imp = imp_arr[i + 1][0]

            if current_imp != next_imp:
                max_amplitude *= self._calc_coefs(current_imp, next_imp)[1]
                sound_speed = imp_arr[i + 1][1]
                wavenumber = self._omega / sound_speed
            wave_anal_sol[i, :] += max_amplitude * np.sin(
                wavenumber * self._x_arr[i] - self._omega * self._t_arr
            )

        # animation
        plt.style.use("dark_background")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        ax1.axis(
            [
                0,
                self.dist_to_observer,
                -self.source_intensity * 1.5,
                self.source_intensity * 1.5,
            ]
        )
        ax2.axis([0, self.dist_to_observer, 0, 100])
        (l,) = ax1.plot([], [])

        # Prepare the data for the max dB value plot
        max_intensity_db = 10 * np.log10(
            np.max(wave_anal_sol[:-1, :], axis=1) / self.i_0
        )
        ax2.plot(self._x_arr[:-1], max_intensity_db)
        ax2.set_xlabel("Position (m)")
        ax2.set_ylabel("Max Intensity (dB)")
        ax2.set_title("Max Intensity (dB) vs Position")

        def myanimation(i):
            l.set_data(self._x_arr, wave_anal_sol[:, i].T)
            ax1.set_title(f"Wave propagation\nTime: {self._t_arr[i]:.4f} s")

        ani = animation.FuncAnimation(
            fig, myanimation, frames=len(self._t_arr), interval=100
        )

        max_intensity = np.max((wave_anal_sol[-2, :]))
        print(
            "Max intensity at observer:",
            10 * np.log10(max_intensity / self.i_0),
            "[dB(SPL)], ",
            max_intensity,
            "[W/m^2]",
        )
        plt.tight_layout()
        ax1.grid(color="r")
        ax2.grid(color="r")

        plt.show()

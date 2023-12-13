import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class RoomModel:
        
    air_z0 = 420  # characteristic impedance of air [PaS/m]
    air_speed = 343  # speed of sound in air [m/s]

    def __init__(self, dist_to_observer: float, source_intensity: float, frequency: float, n_space: int = 50000, n_time: int = 2000, time_to_sim: float = 0.1):
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
            Number of spatial samples, by default 3000
        n_time : int, optional
            Number of temporal samples, by default 200
        time_to_sim : float, optional
            Simulation duration [s], by default 0.1
        """
        self.dist_to_observer = dist_to_observer
        self.n_space = n_space
        self.n_time = n_time

        self.i_0 = 10e-12  # reference value for intensity dB scale [W/m^2]
        self.source_intensity = self.i_0 * 10 ** (source_intensity / 10)  # converted from dB to actual intensity value
        self.omega = 2 * np.pi * frequency

        self.__impedance_val_array = np.full((self.n_space, 2), (self.air_z0, self.air_speed))

        # simulation and plot parameters:
        self.t_arr = np.linspace(0, time_to_sim, n_time)
        self.x_arr = np.linspace(0, dist_to_observer, n_space)

    # private methods
    def __calc_coefs(self, Z_1: float, Z_2: float) -> tuple[float, float]:
        R_i = ((Z_1 - Z_2) ** 2) / ((Z_1 + Z_2) ** 2) # reflection coefficient 
        T_i = 1 - R_i # transmission coefficient 
        return R_i, T_i

    # public methods
    def add_impedant_object(self, start: float, stop: float, char_impedance: float, density: float):
        """Add an object with a certain acoustic impedance. 

        Parameters
        ----------
        start : float
            Start point for object [m]
        stop : float
            Stop point for object [m]
        char_impedance : float
            Characteristic impedance of object [PaS/m]
        density : float
            Density of object [kg/m^3]
        """
        start_pos_index = int(start / self.dist_to_observer * self.n_space)
        stop_pos_index = int(stop / self.dist_to_observer * self.n_space)
        sound_speed = char_impedance / density  # speed of sound in given medium [m/s]

        for i in range(start_pos_index, stop_pos_index):
            self.__impedance_val_array[i][0] = char_impedance
            self.__impedance_val_array[i][1] = sound_speed

    def calc_intensity(self):
        """Simulate system."""
        wave_anal_sol = np.zeros((self.n_space, self.n_time))
        imp_arr = self.__impedance_val_array
        max_amplitude = self.source_intensity
        sound_speed = imp_arr[0][1]  # first value of speed as default
        wavenumber = self.omega / sound_speed

        for i in range(len(imp_arr) - 1):
            current_imp = imp_arr[i][0]
            next_imp = imp_arr[i + 1][0]

            if current_imp != next_imp:
              max_amplitude *= self.__calc_coefs(current_imp, next_imp)[1]
              sound_speed = imp_arr[i + 1][1]
              wavenumber = self.omega / sound_speed
            wave_anal_sol[i, :] += max_amplitude * np.sin(wavenumber * self.x_arr[i] - self.omega * self.t_arr)

        # animation
        plt.style.use("dark_background")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        ax1.axis([0, self.dist_to_observer, -self.source_intensity * 1.5, self.source_intensity * 1.5])
        ax2.axis([0, self.dist_to_observer,0, 100])
        l, = ax1.plot([], [])

        # Prepare the data for the max dB value plot
        max_intensity_db = 10 * np.log10(np.max(wave_anal_sol[:-1, :], axis=1) / self.i_0)
        ax2.plot(self.x_arr[:-1], max_intensity_db)
        ax2.set_xlabel("Position (m)")
        ax2.set_ylabel("Max Intensity (dB)")
        ax2.set_title("Max Intensity (dB) vs Position")

        def myanimation(i):
            l.set_data(self.x_arr, wave_anal_sol[:, i].T)
            ax1.set_title(f"Wave propagation\nTime: {self.t_arr[i]:.4f} s")

        ani = animation.FuncAnimation(fig, myanimation, frames=len(self.t_arr), interval=100)

        max_intensity = np.max((wave_anal_sol[-2, :]))
        print("Max intensity at observer:", 10 * np.log10(max_intensity / self.i_0), "[dB(SPL)], ", max_intensity,"[W/m^2]")
        plt.tight_layout()
        ax1.grid(color='r')
        ax2.grid(color='r')

        plt.show()

def calc_sound_speed(rho: float, K: float) -> float:
    return np.sqrt(K / rho)

def calc_impedance(rho: float, c: float) -> float:
    return rho * c

if __name__ == "__main__":
    rho_door = 696.80 # density of door, densityboard [kg/m^3]
    k_door = 27.65 * 10e6 # elasticity modulus of densityboard. [MPa]

    c_door = calc_sound_speed(rho_door, k_door)
    z_door = calc_impedance(rho_door, c_door)

    sauce_pan_sound_intensity = 83.1 # average intensity of a fork banging against an aluminum saucepan [dB(SPL)]
    sauce_pan_frequency = 2951 # average frequency of a fork banging against an aluminium saucepan [Hz]

    door_pos_start = 1 
    door_pos_end = 1.054
    model_length = door_pos_end + 2 

    model_1 = RoomModel(model_length, sauce_pan_sound_intensity, sauce_pan_frequency)
    model_1.add_impedant_object(door_pos_start, door_pos_end, z_door, rho_door)
    model_1.calc_intensity()

    rho_door = 7850 # density of door, densityboard [kg/m^3]
    k_door = 210000 * 10e6 # elasticity modulus of steel. [MPa]

    c_door = calc_sound_speed(rho_door, k_door)
    z_door = calc_impedance(rho_door, c_door)

    steel_door = RoomModel(model_length, sauce_pan_sound_intensity, sauce_pan_frequency)
    steel_door.add_impedant_object(door_pos_start, door_pos_end, z_door, rho_door)
    steel_door.calc_intensity()

    rho_planks = 390 
    k_planks = 9.45 * 10e6

    c_planks = calc_sound_speed(rho_planks, k_planks)
    z_planks = calc_impedance(rho_planks, c_planks)

    model_1.add_impedant_object(door_pos_end + 0.01, door_pos_end + 0.03, z_planks, rho_planks)
    model_1.calc_intensity()
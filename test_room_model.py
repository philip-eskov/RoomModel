from room_model import * 
import pytest 

# Global test parameters: 

air_z0 = 420  # characteristic impedance of air, PaS/m
air_speed = 343  # speed of sound in air, m/s

dist = 5 # Distance from observer, meters
source_I = 20 # Source intensity, dB(SPL)
f = 1 # Frequency, Hz 
n_space = 5 # Spatial samples 
n_time = 5 # Temporal samples 


def test_add_impedant_object_raises_value_error(): 
  test_model = RoomModel(dist, source_I, f, n_space=n_space, n_time=n_time)
  test_model.add_impedant_object("Test object", 1, 2, 1, 1) 

  with pytest.raises(ValueError): 
   test_model.add_impedant_object("Test object", 1, 2, 1, 1)

def test_remove_impedant_object_raises_value_error(): 
  test_model = RoomModel(dist, source_I, f, n_space=n_space, n_time=n_time)
  test_model.add_impedant_object("Test object", 1, 2, 1, 1) 

  with pytest.raises(ValueError): 
   test_model.remove_impedant_object("Not Test object")


def test_add_impedant_object(): 
  test_model = RoomModel(dist, source_I, f, n_space=n_space, n_time=n_time)
  test_model.add_impedant_object("Test object", 1, 2, 1, 1)

  computed_array = test_model._model_val_array
  expected_array = np.array([[420, 343], [1, 1], [420, 343], [420, 343], [420, 343]])

  msg = f"\n Expected: \n {expected_array} \n Got: \n {computed_array}"

  assert not (computed_array - expected_array).any(), msg 

def test_remove_impedant_object(): 
  test_model = RoomModel(dist, source_I, f, n_space=n_space, n_time=n_time)
  test_model.add_impedant_object("Test object", 1, 2, 1, 1)
  test_model.remove_impedant_object("Test object")

  computed_array = test_model._model_val_array
  expected_array = np.array([[420, 343], [420, 343], [420, 343], [420, 343], [420, 343]])

  msg = f"\n Expected: \n {expected_array} \n Got: \n {computed_array}"

  assert not (computed_array - expected_array).any(), msg 




  
import torch
import math

def test_data(fpl_other_training_set):
    fp = fpl_other_training_set
    assert len(fp) == 18463
    metadata, seq_data = fp[2]
    assert math.isclose(fp.metadata_scaler.inverse_transform(metadata.unsqueeze(0).detach().numpy())[0][0], 1.0, abs_tol=1e-3)
    assert math.isclose(fp.metadata_scaler.inverse_transform(metadata.unsqueeze(0).detach().numpy())[0][3], 4.871517188061858, abs_tol=1e-3)
    assert math.isclose(fp.seq_scaler.inverse_transform(seq_data.unsqueeze(0).detach().numpy())[0], 649, abs_tol=1e-3)
    
    fp._roll(1)
    metadata, seq_data = fp[2]
    assert math.isclose(fp.seq_scaler.inverse_transform(seq_data.unsqueeze(0).detach().numpy())[0], 516, abs_tol=1e-3)
    
    fp._unroll()
    metadata, seq_data = fp[2]
    assert math.isclose(fp.metadata_scaler.inverse_transform(metadata.unsqueeze(0).detach().numpy())[0][0], 1.0, abs_tol=1e-3)
    assert math.isclose(fp.metadata_scaler.inverse_transform(metadata.unsqueeze(0).detach().numpy())[0][3], 4.871517188061858, abs_tol=1e-3)
    assert math.isclose(fp.seq_scaler.inverse_transform(seq_data.unsqueeze(0).detach().numpy())[0], 649, abs_tol=1e-3)
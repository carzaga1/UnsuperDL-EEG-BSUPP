from bscarlos.data.bipolar import BIPOLAR_PAIRS, create_bipolar_channels
from bscarlos.data.edf_ingest import drop_nonrelevant_channels
from bscarlos.testing.synthetic_data import make_synthetic_mne_raw


def test_create_bipolar_channels():
    raw = make_synthetic_mne_raw(n_seconds=5.0, sample_rate=256, seed=0)
    raw = drop_nonrelevant_channels(raw)
    df = raw.to_data_frame()

    bipolar_df = create_bipolar_channels(df)

    assert list(bipolar_df.columns) == [name for name, _, _ in BIPOLAR_PAIRS]
    for name, ch1, ch2 in BIPOLAR_PAIRS:
        expected = df[ch1] - df[ch2]
        assert (bipolar_df[name] == expected).all()

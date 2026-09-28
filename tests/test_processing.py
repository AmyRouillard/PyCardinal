import numpy as np
import pytest

from pycardinal import (
    MSImagingArrays,
    PositionDataFrame,
    bin_spectra,
    estimate_domain,
    estimate_reference_mz,
    normalize,
    peak_align,
    peak_pick,
    peak_process,
    process,
    recalibrate,
    reduce_baseline,
    simulate_image,
    simulate_spectra,
    smooth,
)


def make_ragged() -> MSImagingArrays:
    return MSImagingArrays(
        mz=[np.array([100.0, 200.0]), np.array([100.0, 200.0])],
        intensity=[np.array([2.0, 2.0]), np.array([1.0, 3.0])],
        pixel_data=PositionDataFrame({"x": [1, 2], "y": [1, 1]}),
    )


def test_tic_normalization_is_deferred_and_scales_each_spectrum() -> None:
    source = make_ragged()
    queued = normalize(source, method="tic", scale=1.0)

    assert len(queued.processing) == 1
    np.testing.assert_array_equal(source.intensity[0], [2.0, 2.0])

    result = process(queued)

    np.testing.assert_allclose(
        [values.sum() for values in result.intensity], [1.0, 1.0]
    )
    assert result.processing == []


def test_rms_normalization_uses_requested_target() -> None:
    result = process(normalize(make_ragged(), method="rms", scale=3.0))

    np.testing.assert_allclose(
        [np.sqrt(np.mean(values**2)) for values in result.intensity],
        [3.0, 3.0],
    )


def test_reference_normalization_scales_matching_mz() -> None:
    result = process(
        normalize(
            make_ragged(),
            method="reference",
            ref=[100.0],
            scale=1.0,
            tolerance=5,
            units="mz",
        )
    )

    np.testing.assert_allclose(result.intensity[0], [1.0, 1.0])
    np.testing.assert_allclose(result.intensity[1], [1.0, 3.0])


def make_profile() -> MSImagingArrays:
    spectrum = simulate_spectra(
        npeaks=6,
        by=20_000,
        baseline=2,
        random_state=101,
    )
    return MSImagingArrays(
        mz=[spectrum["mz"].to_numpy()],
        intensity=[spectrum["intensity"].to_numpy()],
        pixel_data=PositionDataFrame({"x": [1], "y": [1]}),
    )


@pytest.mark.parametrize(
    "method",
    ["gaussian", "bilateral", "adaptive", "diff", "guide", "pag", "sgolay", "ma"],
)
def test_smoothing_methods_are_deferred_and_preserve_spectrum_shape(
    method: str,
) -> None:
    source = make_profile()
    queued = smooth(source, method=method, width=5)

    assert len(queued.processing) == 1
    result = process(queued)

    assert len(result.intensity[0]) == len(source.intensity[0])
    assert np.isfinite(result.intensity[0]).all()
    np.testing.assert_array_equal(source.intensity[0], make_profile().intensity[0])


@pytest.mark.parametrize("method", ["locmin", "hull", "snip", "median"])
def test_baseline_methods_are_deferred_and_nonnegative(method: str) -> None:
    source = make_profile()
    result = process(reduce_baseline(source, method=method, window=9))

    assert len(result.intensity[0]) == len(source.intensity[0])
    assert np.isfinite(result.intensity[0]).all()
    assert np.all(result.intensity[0] >= 0)


@pytest.mark.parametrize("method", ["locmax", "dtw", "cow"])
def test_recalibration_moves_peak_to_reference(method: str) -> None:
    source = MSImagingArrays(
        mz=[np.array([98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0])],
        intensity=[np.array([0.0, 0.0, 1.0, 10.0, 1.0, 0.0, 0.0])],
        pixel_data=PositionDataFrame({"x": [1], "y": [1]}),
    )

    result = process(
        recalibrate(
            source,
            ref=[102.0],
            method=method,
            tolerance=2.0,
            units="mz",
        )
    )

    assert result.mz[0][3] == 102.0
    np.testing.assert_array_equal(result.intensity[0], source.intensity[0])


def make_peak_arrays() -> MSImagingArrays:
    return MSImagingArrays(
        mz=[
            np.array([99.0, 100.0, 101.0, 199.0, 200.0, 201.0]),
            np.array([99.1, 100.1, 101.1, 199.1, 200.1, 201.1]),
        ],
        intensity=[
            np.array([0.0, 8.0, 0.0, 0.0, 6.0, 0.0]),
            np.array([0.0, 7.0, 0.0, 0.0, 5.0, 0.0]),
        ],
        pixel_data=PositionDataFrame({"x": [1, 2], "y": [1, 1]}),
        centroided=True,
    )


def test_peak_pick_queues_and_extracts_local_maxima() -> None:
    source = make_peak_arrays()
    source.centroided = False
    queued = peak_pick(source, method="sd", snr=0.1)

    assert len(queued.processing) == 1
    assert queued.centroided is True
    assert source.centroided is False
    result = process(queued)

    assert [len(values) for values in result.mz] == [2, 2]
    np.testing.assert_allclose(result.mz[0], [100.0, 200.0])
    np.testing.assert_allclose(result.intensity[1], [7.0, 5.0])


def test_reference_peak_pick_extracts_matching_peaks() -> None:
    source = make_peak_arrays()
    result = process(
        peak_pick(
            source,
            ref=[100.0, 200.0],
            tolerance=1.0,
            units="mz",
        )
    )

    np.testing.assert_allclose(result.mz[1], [100.0, 200.0])
    np.testing.assert_allclose(result.intensity[1], [7.0, 5.0])


def test_peak_alignment_returns_shared_sparse_features() -> None:
    picked = process(peak_pick(make_peak_arrays(), method="sd", snr=0.1))
    aligned = peak_align(
        picked,
        ref=[100.0, 200.0],
        method="sd",
        snr=0.1,
        tolerance=1.0,
        units="mz",
    )

    assert aligned.shape == (2, 2)
    assert aligned.centroided is True
    np.testing.assert_allclose(aligned.mz, [100.0, 200.0])
    np.testing.assert_allclose(aligned.intensity.toarray(), [[8.0, 7.0], [6.0, 5.0]])
    assert aligned.feature_data["count"].tolist() == [2, 2]


def test_peak_alignment_infers_tolerance_without_merging_neighboring_peaks() -> None:
    aligned = peak_align(
        make_peak_arrays(),
        method="sd",
        snr=0.1,
        units="mz",
    )

    assert aligned.shape == (2, 2)
    np.testing.assert_allclose(aligned.mz, [100.05, 200.05])


def test_peak_process_combines_pick_align_and_frequency_filter() -> None:
    result = peak_process(
        make_peak_arrays(),
        ref=[100.0, 200.0],
        method="sd",
        snr=0.1,
        tolerance=1.0,
        units="mz",
        filter_freq=True,
    )

    assert result.shape == (2, 2)
    np.testing.assert_allclose(result.mz, [100.0, 200.0])


def test_peak_process_can_estimate_reference_from_sampled_spectra() -> None:
    result = peak_process(
        make_peak_arrays(),
        method="sd",
        snr=0.1,
        sample_size=1,
        units="mz",
    )

    assert result.shape == (2, 2)
    np.testing.assert_allclose(result.mz, [100.0, 200.0])


def test_phase3_preprocessing_and_peak_pipeline_on_simulated_image() -> None:
    source = simulate_image(
        preset=1,
        npeaks=8,
        dim=(4, 3),
        by=20_000,
        jitter=False,
        random_state=303,
    )
    queued = normalize(source, method="tic", scale=1.0)
    queued = smooth(queued, method="gaussian", width=3)
    queued = reduce_baseline(queued, method="locmin", window=5)
    result = peak_process(
        queued,
        method="sd",
        snr=0.2,
        filter_freq=False,
    )

    assert result.shape[1] == len(source)
    assert result.centroided is True
    assert np.isfinite(result.intensity.data).all()
    assert np.all(result.intensity.data >= 0)


def test_binning_and_domain_estimation_on_simulated_axes() -> None:
    source = MSImagingArrays(
        mz=[np.array([100.1, 100.9]), np.array([100.4, 100.8])],
        intensity=[np.array([2.0, 4.0]), np.array([6.0, 8.0])],
        pixel_data=PositionDataFrame({"x": [1, 2], "y": [1, 1]}),
    )
    axis = estimate_domain(source.mz, units="absolute")
    assert axis[0] == 100.1
    binned = bin_spectra(source, ref=[100.0, 101.0], units="mz", tolerance=0.5)

    assert binned.shape == (2, 2)
    np.testing.assert_allclose(binned.intensity.toarray(), [[2.0, 6.0], [4.0, 8.0]])


def test_estimate_reference_mz_reuses_shared_experiment_axis() -> None:
    image = simulate_image(
        preset=1,
        npeaks=6,
        dim=(3, 2),
        by=10_000,
        jitter=False,
        random_state=808,
    )

    np.testing.assert_array_equal(estimate_reference_mz(image), image.mz)


@pytest.mark.parametrize(
    ("method", "expected"),
    [("sum", 6.0), ("mean", 3.0), ("max", 4.0), ("min", 2.0)],
)
def test_binning_aggregations_share_the_same_tolerance(
    method: str,
    expected: float,
) -> None:
    source = MSImagingArrays(
        mz=[np.array([100.1, 100.2])],
        intensity=[np.array([2.0, 4.0])],
        pixel_data=PositionDataFrame({"x": [1], "y": [1]}),
    )

    result = bin_spectra(
        source,
        ref=[100.0],
        method=method,
        units="mz",
        tolerance=0.5,
    )

    assert result.intensity[0, 0] == expected

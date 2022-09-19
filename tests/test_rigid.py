import numpy as np
import pytest
from cv2 import perspectiveTransform

import endeform.interpolation.rigid as rigid

RTOL = 1e-5
ATOL = 0
# comparisons check for differences of less than ATOL + RTOL*abs(DESIRED)

# Some points
N_points = 20
h, w, d = 600, 800, 256
X_2D = np.random.rand(N_points, 2) * np.array([[h, w]])
X_3D = np.random.rand(N_points, 3) * np.array([[h, w, d]])

# Translations
trans_2D = np.array([[-0.1 * h, 0.07 * w]])
trans_3D = np.array([[-0.1 * h, 0.07 * w, 0.11 * d]])

# Rotations in 2D and 3D
phi0, phi1, phi2 = np.pi / 10, np.pi / 14, -np.pi / 8
rot2 = np.array([[np.cos(phi0), -np.sin(phi0)], [np.sin(phi0), np.cos(phi0)]])
rot3 = (
    np.array(
        [[np.cos(phi0), -np.sin(phi0), 0], [np.sin(phi0), np.cos(phi0), 0], [0, 0, 1]]
    )
    @ np.array(
        [[np.cos(phi1), 0, np.sin(phi1)], [0, 1, 0], [-np.sin(phi1), 0, np.cos(phi1)]]
    )
    @ np.array(
        [[1, 0, 0], [0, np.cos(phi2), -np.sin(phi2)], [0, np.sin(phi2), np.cos(phi2)]]
    )
)

# Scalings
scale0, scale1, scale2 = 1.1, -0.8, 0.75

# Purely translated points
Y_2D_translated = X_2D + trans_2D
Y_3D_translated = X_3D + trans_3D

# Procrustes-transformed points
Y_2D_Procrustes = scale0 * (X_2D @ rot2) + trans_2D
# Points with anisotropic scaling, rotation, and translation (that should be fully affine)
Y_2D_anisotropic_scaling = (X_2D * np.array([[scale0, scale1]])) @ rot2 + trans_2D
# Points undergoing a full homography transform (only lines are preserved)
H = np.random.rand(3, 3)
H = H / H[2, 2]  #  Normalize to (3,3) element ==1, now it's a homography transformation
Y_2D_homography = perspectiveTransform(X_2D[np.newaxis, ...], H).squeeze()
# extra singleton dimension necessary because cv2 is stupid here and expects the n-th datapoint at X[0, n, :]


@pytest.mark.parametrize("norm", [1, 2])
class TestTranslation:
    """Test the TranslationInterpolator for 2D and 3D inputs, by using inputs that are indeed purely translated"""

    def test_TranslationInterpolator2D_fit(self, norm):
        interp = rigid.TranslationInterpolator(norm=norm)
        interp.fit(X_2D, Y_2D_translated)
        np.testing.assert_allclose(
            interp.coef_, np.vstack((np.eye(2), trans_2D)), rtol=RTOL, atol=ATOL
        )

    def test_TranslationInterpolator2D_eval(self, norm):
        interp = rigid.TranslationInterpolator(norm=norm)
        interp.fit(X_2D, Y_2D_translated)
        np.testing.assert_allclose(
            Y_2D_translated, interp.eval(X_2D), rtol=RTOL, atol=ATOL
        )

    def test_TranslationInterpolator3D_fit(self, norm):
        interp = rigid.TranslationInterpolator(norm=norm)
        interp.fit(X_3D, Y_3D_translated)
        np.testing.assert_allclose(
            interp.coef_, np.vstack((np.eye(3), trans_3D)), rtol=RTOL, atol=ATOL
        )

    def test_TranslationInterpolator3D_eval(self, norm):
        interp = rigid.TranslationInterpolator(norm=norm)
        interp.fit(X_3D, Y_3D_translated)
        np.testing.assert_allclose(
            Y_3D_translated, interp.eval(X_3D), rtol=RTOL, atol=ATOL
        )


class TestCV2RigidTransforms:
    """Test the rigid interpolators that are wrapped from cv2"""

    interp_procrust = rigid.ProcrustesInterpolator()
    interp_aff = rigid.AffineInterpolator()
    interp_hom = rigid.HomographyInterpolator()

    @pytest.mark.parametrize("Y_to", [Y_2D_translated, Y_2D_Procrustes])
    def test_ProkrustesInterpolator_fit_eval(self, Y_to):
        self.interp_procrust.fit(X_2D, Y_to)
        # np.testing.assert_allclose(interp.coef_, np.vstack( (np.eye(2), trans_2D) ) )
        np.testing.assert_allclose(
            Y_to, self.interp_procrust.eval(X_2D), rtol=RTOL, atol=ATOL
        )

    @pytest.mark.parametrize(
        "Y_to", [Y_2D_translated, Y_2D_Procrustes, Y_2D_anisotropic_scaling]
    )
    def test_AffineInterpolator_fit_eval(self, Y_to):
        self.interp_aff.fit(X_2D, Y_to)
        # np.testing.assert_allclose(interp.coef_, np.vstack( (np.eye(2), trans_2D) ) )
        np.testing.assert_allclose(
            Y_to, self.interp_aff.eval(X_2D), rtol=RTOL, atol=ATOL
        )

    @pytest.mark.parametrize(
        "Y_to", [Y_2D_translated, Y_2D_Procrustes, Y_2D_homography]
    )
    def test_HomographyInterpolator_fit_eval(self, Y_to):
        self.interp_hom.fit(X_2D, Y_to)
        # np.testing.assert_allclose(interp.coef_, np.vstack( (np.eye(2), trans_2D) ) )
        np.testing.assert_allclose(
            Y_to, self.interp_hom.eval(X_2D), rtol=RTOL, atol=ATOL
        )

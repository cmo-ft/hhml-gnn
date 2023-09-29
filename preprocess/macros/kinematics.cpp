// some key physics value
const float GeV = 1e3; // MeV to GeV
const float Z_Mass = 91.1876 * GeV; // MeV


int low_mass_cut(int Evt_SFOS, float m01, float m02) {
    int result = 1;

    if (Evt_SFOS == 1) result = (m01 > 12. * GeV);
    if (Evt_SFOS == 2) result = (m02 > 12. * GeV);
    if (Evt_SFOS == 3) result = ((m01 > 12. * GeV) && (m02 > 12. * GeV));

    return result;
}

int z_mass_cut(int Evt_SFOS, float m01, float m02) {
    int result = 1;

    float Zm_1 = fabs(m01 - Z_Mass);
    float Zm_2 = fabs(m02 - Z_Mass);

    if (Evt_SFOS == 1) result = (Zm_1 > 10. * GeV);
    if (Evt_SFOS == 2) result = (Zm_2 > 10. * GeV);
    if (Evt_SFOS == 3) result = ((Zm_1 > 10. * GeV) && (Zm_2 > 10. * GeV));

    return result;
}

int m_lll_cut(float m_lll) {
    return fabs(m_lll - Z_Mass) >= 10. * GeV;
}
int flavor_category(Float_t lep_ID_0, Float_t lep_ID_1, Float_t lep_ID_2) {
    int FlavorCategory = -1;

    // Sanity Check
    int nElectron = 0;
    int nMuon = 0;

    std::vector<Float_t> lep_ID = {lep_ID_0, lep_ID_1, lep_ID_2};
    for (auto id: lep_ID) {
        if (abs(id) == 11) nElectron++;
        if (abs(id) == 13) nMuon++;
    }

    if (nElectron + nMuon != 3) return -1; // not belong to 3-lepton

    if (nElectron == 3) FlavorCategory = 1; // eee
    if (nElectron == 2 && abs(lep_ID[0]) == 11 && abs(lep_ID[1]) == 11) FlavorCategory = 2; // eem
    if (nElectron == 2 && abs(lep_ID[0]) == 11 && abs(lep_ID[1]) == 13) FlavorCategory = 3; // eme
    if (nElectron == 1 && abs(lep_ID[0]) == 11) FlavorCategory = 4; // emm
    if (nElectron == 2 && abs(lep_ID[0]) == 13) FlavorCategory = 5; // mee
    if (nElectron == 1 && abs(lep_ID[0]) == 13 && abs(lep_ID[1]) == 11) FlavorCategory = 6; // mem
    if (nElectron == 1 && abs(lep_ID[0]) == 13 && abs(lep_ID[1]) == 13) FlavorCategory = 7; // mme
    if (nElectron == 0) FlavorCategory = 8; // mmm
    /*  SFOS0 = 4 && 5;
     *  SFSO1 = 2 && 3 && 6 && 7;
     *  SFSO2 = 1 && 8;
     *  */


    return FlavorCategory;
}

int sfos_map(int FlavorCategory) {
    if (FlavorCategory == 4 || FlavorCategory == 5) return 0; // no SFOS
    if (FlavorCategory == 2 || FlavorCategory == 7) return 1; // SFOS-1: (0,1)
    if (FlavorCategory == 3 || FlavorCategory == 6) return 2; // SFOS-1: (0,2)
    if (FlavorCategory == 1 || FlavorCategory == 8) return 3; // SFOS-2: (0,1) && (0,2)

    return -1;
}




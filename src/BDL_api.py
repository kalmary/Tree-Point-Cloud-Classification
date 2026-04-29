import requests
import math
from collections import Counter
from typing import Optional

SPECIES_MODEL = {
    0: ['Pinus', 'sosna'],
    1: ['Picea', 'świerk'],
    2: ['Abies', 'jodła'],
    3: ['Larix', 'modrzew'],
    4: ['Pseudotsuga', 'daglezja'],
    5: ['Quercus', 'dąb'],
    6: ['Ulmus', 'wiąz'],
    7: ['Fagus', 'buk'],
    8: ['Tilia', 'lipa'],
    9: ['Carpinus', 'grab'],
    10: ['Acer', 'klon'],
    11: ['Fraxinus', 'jesion'],
    12: ['Betula', 'brzoza'],
    13: ['Corylus', 'leszczyna'],
    14: ['Crataegus', 'głóg'],
    15: ['Others', 'Inne'],
    16: ['Incorrect segmentation', 'Błędna segmentacja']
}    

SPECIES_DBL = {
    "BRZ":  ["Betula_pendula",         "Brzoza brodawkowata"],
    "BK":   ["Fagus_sylvatica",        "Buk zwyczajny"],
    "DB.S": ["Quercus_robur",          "Dąb szypułkowy"],
    "DB.B": ["Quercus_petraea",        "Dąb bezszypułkowy"],
    "DB.C": ["Quercus_rubra",          "Dąb czerwony"],
    "GB":   ["Carpinus_betulus",       "Grab pospolity"],
    "JS":   ["Fraxinus_excelsior",     "Jesion wyniosły"],
    "KL":   ["Acer_platanoides",       "Klon pospolity"],
    "JW":   ["Acer_pseudoplatanus",    "Klon jawor"],
    "KL.P": ["Acer_campestre",         "Klon polny"],
    "LP":   ["Tilia_cordata",          "Lipa drobnolistna"],
    "WZ.S": ["Ulmus_laevis",           "Wiąz szypułkowy"],
    "GŁG":  ["Crataegus_monogyna",     "Głóg jednoszyjkowy"],
    "LSZ":  ["Corylus_avellana",       "Leszczyna pospolita"],
    "DG":   ["Pseudotsuga_menziesii",  "Daglezja zielona"],
    "JD":   ["Abies_alba",             "Jodła pospolita"],
    "MD":   ["Larix_decidua",          "Modrzew europejski"],
    "SO":   ["Pinus_sylvestris",       "Sosna zwyczajna"],
    "ŚW":   ["Picea_abies",            "Świerk pospolity"],
    "OS":   ["Populus_tremula",        "Topola osika"],
    "TP":   ["Populus_alba",           "Topola biała"],
    "TP.C": ["Populus_nigra",          "Topola czarna"],
    "DB":   ["Quercus species",        "Dąb nieokreślony"]
}

RDLP_TO_COLLECTION = {
    "BIAŁYSTOK":    "RDLP_Bialystok_wydzielenia",
    "KATOWICE":     "RDLP_Katowice_wydzielenia",
    "KRAKÓW":       "RDLP_Krakow_wydzielenia",
    "KROSNO":       "RDLP_Krosno_wydzielenia",
    "LUBLIN":       "RDLP_Lublin_wydzielenia",
    "ŁÓDŹ":         "RDLP_Lodz_wydzielenia",
    "OLSZTYN":      "RDLP_Olsztyn_wydzielenia",
    "PIŁA":         "RDLP_Pila_wydzielenia",
    "POZNAŃ":       "RDLP_Poznan_wydzielenia",
    "SZCZECIN":     "RDLP_Szczecin_wydzielenia",
    "SZCZECINEK":   "RDLP_Szczecinek_wydzielenia",
    "TORUŃ":        "RDLP_Torun_wydzielenia",
    "WROCŁAW":      "RDLP_Wroclaw_wydzielenia",
    "ZIELONA GÓRA": "RDLP_Zielona_Gora_wydzielenia",
    "GDAŃSK":       "RDLP_Gdansk_wydzielenia",
    "RADOM":        "RDLP_Radom_wydzielenia",
    "WARSZAWA":     "RDLP_Warszawa_wydzielenia",
}

class BDLCall():

    def __init__(
        self,
        species_dbl: dict = SPECIES_DBL,
        species_model: dict = SPECIES_MODEL,
        rdlp_dict: dict = RDLP_TO_COLLECTION,
        size_m: int = 600,
        model_based: bool = True,
    ):
        self.session = requests.Session()
        self.base = "https://ogcapi.bdl.lasy.gov.pl"
        self.species_dbl = species_dbl
        self.species_model = species_model
        self.rdlp_dict = rdlp_dict
        self.size_m = size_m
        self.model_based = model_based


    def _fetch(self, url: str, params: Optional[dict] = None) -> dict:
        r = self.session.get(url, params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    def _fetch_all(self, collection: str, bbox: str) -> list:
        feats = []
        url = f"{self.base}/collections/{collection}/items"
        params = {"bbox": bbox, "limit": 1000, "f": "json"}
        while url:
            data = self._fetch(url, params)
            feats.extend(data.get("features", []))
            url = next(
                (l["href"] for l in data.get("links", []) if l.get("rel") == "next"),
                None,
            )
            params = None
        return feats


    def _bbox_meters(self, lat: float, lon: float) -> str:
        half = self.size_m / 2
        dlat = half / 111_320
        dlon = half / (111_320 * math.cos(math.radians(lat)))
        return f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}"

    def _get_rdlp_collection(self, lat: float, lon: float) -> Optional[str]:
        url = f"{self.base}/collections/rdlp/items"
        params = {"bbox": f"{lon},{lat},{lon},{lat}", "f": "json"}
        feats = self._fetch(url, params).get("features", [])
        if not feats:
            return None
        region = feats[0]["properties"].get("region_name")
        return self.rdlp_dict.get(region)
    

    def _count_species_in_area(self, lat: float, lon: float) -> Counter:
        collection = self._get_rdlp_collection(lat, lon)
        if collection is None:
            return Counter()

        bbox = self._bbox_meters(lat, lon)
        feats = self._fetch_all(collection, bbox)

        counts = Counter()
        for f in feats:
            sp_code = f["properties"].get("species_cd")
            entry = self.species_dbl.get(sp_code)
            if entry is None:
                continue
            counts[entry[0]] += 1
        return counts

    @staticmethod
    def _most_common(counts: Counter) -> Optional[str]:
        return counts.most_common(1)[0][0] if counts else None

    def _most_common_in_genus(self, counts: Counter, genus_latin: str) -> Optional[str]:
        """Najczęstszy gatunek z danego rodzaju w obszarze (None jeśli brak)."""
        filtered = Counter({
            sp: n for sp, n in counts.items() if sp.startswith(genus_latin + "_")
        })
        return self._most_common(filtered)

    def _default_species_for_genus(self, genus_latin: str) -> Optional[str]:
        """Domyślny (pierwszy w SPECIES_DBL) gatunek dla danego rodzaju."""
        for latin, _ in self.species_dbl.values():
            if latin.startswith(genus_latin + "_"):
                return latin
        return None


    def find_species(self, lat: float, lon: float, input_class: int) -> str:
        genus_latin = self.species_model[input_class][0]

        counts = self._count_species_in_area(lat, lon)

        if genus_latin in ("Others", "Incorrect segmentation"):
            return self._most_common(counts) or genus_latin

        # zwykły rodzaj — np. "Acer"
        in_area = self._most_common_in_genus(counts, genus_latin)
        if in_area is not None:
            return in_area

        # rodzaju brak w obszarze
        if self.model_based:
            return self._default_species_for_genus(genus_latin) or genus_latin
        return self._most_common(counts) or genus_latin


if __name__ == "__main__":
    bdl = BDLCall(size_m=10000)
    print(bdl.find_species(53.614462, 22.487602, 0))

    # size_m powinno być w init 
    # w find_species powinna być klasa podana jako int
    # int zwrócony przez model -> 
    # -> szukasz obszaru na którym jest dane drzewo 
    # -> patrzysz jakie drzewa występują na tym terenie
    # -> robisz dopasowanie:
    # --> jeśli masz model zwrócił np. klon: szukasz jakie klony występują na tym obszarze i zwracasz najczęstszy gatunek klona
    # --> jeśli na tym obszarze nie ma np. klonu a model go zwrócił działasz zależnie od flagi: 
    # ---> model_based: bool = True: zwracasz klon (ten który w Polsce jest częstszy, dokładana nazwa)
    # ---> model_based: bool = False: zwracasz najczęstszy gatunek z obszaru
    # --> jeśli model zwrócił others 15: zwracasz najczęstsze drzewo z obszaru
    # --> jeśli model zwrócił others 15, ale nie masz danych o obszarze: zwracasz others
    # --> tak samo jak others działa błędna segmentacja

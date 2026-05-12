import requests
import math
from collections import Counter
from typing import Optional
from pyproj import Transformer
import numpy as np

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

SPECIES_BDL = {
    "BRZ":  ["Betula_pendula",         "Brzoza brodawkowata",   0],
    "BK":   ["Fagus_sylvatica",        "Buk zwyczajny",         1],
    "DB":   ["Quercus_species",        "Dąb nieokreślony",      2],
    "DB.S": ["Quercus_robur",          "Dąb szypułkowy",        3],
    "DB.B": ["Quercus_petraea",        "Dąb bezszypułkowy",     4],
    "DB.C": ["Quercus_rubra",          "Dąb czerwony",          5],
    "GB":   ["Carpinus_betulus",       "Grab pospolity",        6],
    "JS":   ["Fraxinus_excelsior",     "Jesion wyniosły",       7],
    "KL":   ["Acer_platanoides",       "Klon pospolity",        8],
    "JW":   ["Acer_pseudoplatanus",    "Klon jawor",            9],
    "KL.P": ["Acer_campestre",         "Klon polny",           10],
    "LP":   ["Tilia_cordata",          "Lipa drobnolistna",    11],
    "WZ.S": ["Ulmus_laevis",           "Wiąz szypułkowy",      12],
    "GŁG":  ["Crataegus_monogyna",     "Głóg jednoszyjkowy",   13],
    "LSZ":  ["Corylus_avellana",       "Leszczyna pospolita",  14],
    "DG":   ["Pseudotsuga_menziesii",  "Daglezja zielona",     15],
    "JD":   ["Abies_alba",             "Jodła pospolita",      16],
    "MD":   ["Larix_decidua",          "Modrzew europejski",   17],
    "SO":   ["Pinus_sylvestris",       "Sosna zwyczajna",      18],
    "ŚW":   ["Picea_abies",            "Świerk pospolity",     19],
    "OS":   ["Populus_tremula",        "Topola osika",         20],
    "TP":   ["Populus_alba",           "Topola biała",         21],
    "TP.C": ["Populus_nigra",          "Topola czarna",        22],
    "O_S":  ["Others",                 "Inne",                 23],
    "I_S":  ["Incorrect segmentation", "Błędna segmentacja",   24]
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
        species_dbl: dict = SPECIES_BDL,
        species_model: dict = SPECIES_MODEL,
        rdlp_dict: dict = RDLP_TO_COLLECTION,
        size_m: int = 5000,
        model_based: bool = True,
    ):
        self.session = requests.Session()
        self.base = "https://ogcapi.bdl.lasy.gov.pl"
        self.species_dbl = species_dbl
        self.species_model = species_model
        self.rdlp_dict = rdlp_dict
        self.size_m = size_m
        self.model_based = model_based
        self._latin_to_int = {val[0]: val[2] for val in species_dbl.values()}
        self._species_storage: Optional[Counter] = None
        self._species_tile_map: Optional[dict[tuple[int, int], Counter]] = None
        self._tile_origin_xy: Optional[np.ndarray] = None


    def _fetch(self, url: str, params: Optional[dict] = None) -> dict:
        # Performs a single GET request to the BDL API and returns the parsed JSON response.
        r = self.session.get(url, params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    def _fetch_all(self, collection: str, bbox: str) -> list:
        # Fetches all features from a collection within a bbox, following pagination links until exhausted.
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

    @staticmethod
    def utm_to_latlon(easting: float, northing: float, crs=None):
  
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(easting, northing)
        return lat, lon

    def _bbox_meters(self, lat: float, lon: float) -> str:
        # Builds a bbox string of size_m × size_m meters centered on the given lat/lon coordinates.
        half = self.size_m / 2
        dlat = half / 111_320
        dlon = half / (111_320 * math.cos(math.radians(lat)))
        return f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}"

    def _get_rdlp_collection(self, lat: float, lon: float) -> Optional[str]:
        # Resolves the RDLP region containing the point and returns its corresponding wydzielenia collection name.
        url = f"{self.base}/collections/rdlp/items"
        params = {"bbox": f"{lon},{lat},{lon},{lat}", "f": "json"}
        feats = self._fetch(url, params).get("features", [])
        if not feats:
            return None
        region = feats[0]["properties"].get("region_name")
        return self.rdlp_dict.get(region)
    

    def _count_species_in_area(self, lat: float, lon: float) -> Counter:
        # Counts occurrences of each known species in forest stands intersecting the bbox around the point.
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
    def _validate_pcd(pcd: np.ndarray) -> None:
        if pcd.ndim != 2 or pcd.shape[1] < 2:
            raise ValueError("pcd must be a 2D array with at least x and y columns")
        if len(pcd) == 0:
            raise ValueError("pcd must not be empty")

    @staticmethod
    def _xy_bounds(pcd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy_min = pcd[:, :2].min(axis=0)
        xy_max = pcd[:, :2].max(axis=0)
        return xy_min, xy_max

    def _counts_for_xy(self, xy: np.ndarray, crs) -> Counter:
        lat, lon = self.utm_to_latlon(xy[0], xy[1], crs=crs)
        return self._count_species_in_area(lat, lon)

    def _build_single_species_storage(
        self,
        xy_min: np.ndarray,
        xy_max: np.ndarray,
        crs,
    ) -> Counter:
        center_xy = (xy_min + xy_max) / 2
        return self._counts_for_xy(center_xy, crs)

    def _tile_id_for_xy(self, xy: np.ndarray) -> tuple[int, int]:
        tile_xy = np.floor((xy - self._tile_origin_xy) / self.size_m).astype(int)
        return int(tile_xy[0]), int(tile_xy[1])

    def _tile_center_xy(self, tile_id: tuple[int, int]) -> np.ndarray:
        return self._tile_origin_xy + (np.asarray(tile_id, dtype=float) + 0.5) * self.size_m

    def _build_species_tile_map(self, pcd: np.ndarray, xy_min: np.ndarray, crs) -> dict[tuple[int, int], Counter]:
        self._tile_origin_xy = xy_min
        tile_ids = {
            self._tile_id_for_xy(xy)
            for xy in pcd[:, :2]
        }
        return {
            tile_id: self._counts_for_xy(self._tile_center_xy(tile_id), crs)
            for tile_id in tile_ids
        }

    def prepare_species_storage(self, pcd: np.ndarray, crs):
        # Builds one shared counter or a non-overlapping tile map for a vegetation cloud.
        self.clear_species_storage()
        self._validate_pcd(pcd)

        xy_min, xy_max = self._xy_bounds(pcd)
        xy_size = xy_max - xy_min

        if np.any(xy_size > self.size_m):
            self._species_tile_map = self._build_species_tile_map(pcd, xy_min, crs)
            return self._species_tile_map

        self._species_storage = self._build_single_species_storage(xy_min, xy_max, crs)
        return self._species_storage

    def clear_species_storage(self) -> None:
        self._species_storage = None
        self._species_tile_map = None
        self._tile_origin_xy = None

    def _get_stored_counts(self, pcd: np.ndarray) -> Optional[Counter]:
        if self._species_storage is not None:
            return self._species_storage

        if self._species_tile_map is None:
            return None

        centroid = pcd[:, :2].mean(axis=0)
        tile_id = self._tile_id_for_xy(centroid)
        return self._species_tile_map.get(tile_id)

    def _most_common(self, counts: Counter) -> Optional[int]:
        # Returns the species int code of the most frequent entry in counts, or None if empty.
        return self._get_int(counts.most_common(1)[0][0]) if counts else None

    def _most_common_in_genus(self, counts: Counter, genus_latin: str) -> Optional[int]:
        # Returns the species int code of the most frequent species in the given genus, or None if absent.
        filtered = Counter({
            sp: n for sp, n in counts.items() if sp.startswith(genus_latin + "_")
        })
        return self._most_common(filtered)

    def _default_species_for_genus(self, genus_latin: str) -> int:
        # Returns the int code of the default (most common in Poland) species for a genus, falling back to "Others".
        for val in self.species_dbl.values():
            if val[0].startswith(genus_latin + "_"):
                return val[2]
        return self._get_int("Others")
    
    def _get_int(self, latin_name: str) -> int:
        # Looks up the integer code assigned to a given Latin species name in SPECIES_DBL.
        return self._latin_to_int[latin_name]

    def predict(self, pcd: np.ndarray, crs, tree_label: int) -> int:
        stored_counts = self._get_stored_counts(pcd)
        if stored_counts is not None:
            return self._resolve_species(stored_counts, tree_label)

        centroid = pcd.mean(axis=0)
        lat, lon = self.utm_to_latlon(centroid[0], centroid[1], crs=crs)
        tree_label = self.find_species(lat, lon, tree_label)
        return tree_label

    def find_species(self, lat: float, lon: float, input_class: int) -> int:
        # Maps a model-predicted genus class to a specific species int code using BDL area data and fallback rules.
        counts = self._count_species_in_area(lat, lon)
        return self._resolve_species(counts, input_class)

    def _resolve_species(self, counts: Counter, input_class: int) -> int:
        genus_latin = self.species_model[input_class][0]

        if genus_latin == "Incorrect segmentation":
            return self._get_int(genus_latin)

        if genus_latin == "Others":
            return self._most_common(counts) or self._get_int(genus_latin)

        in_area = self._most_common_in_genus(counts, genus_latin)
        if in_area is not None:
            return in_area

        genus_default = self._default_species_for_genus(genus_latin)
        
        if self.model_based:
            return genus_default
        return self._most_common(counts) or genus_default


if __name__ == "__main__":
    bdl = BDLCall(size_m=5000, model_based=True)
    print(bdl.find_species(53.643773,22.465687, 15))

    # size_m powinno być w init 
    # w find_species powinna być klasa podana jako int
    # output find_species to int (key z SPECIES_DBL)
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
    # --> błędna segmentacja zawsze zwraca błędną segmentację

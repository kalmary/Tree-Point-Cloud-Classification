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

SPECIES_DBL = {
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
        species_dbl: dict = SPECIES_DBL,
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

        # tile_map: (ix, iy) -> Counter[latin_name, count]
        # tile origin (meters, same CRS as the PCD passed to build_data_map)
        self._tile_map: Optional[dict] = None
        self._tile_origin: Optional[tuple] = None   # (x_min, y_min)
        self._tile_crs = None

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def utm_to_latlon(easting: float, northing: float, crs=None):
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(easting, northing)
        return lat, lon

    def _bbox_meters(self, lat: float, lon: float) -> str:
        half = self.size_m / 2
        dlat = half / 111_320
        dlon = half / (111_320 * math.cos(math.radians(lat)))
        return f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}"

    # ------------------------------------------------------------------
    # BDL lookup helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Species resolution helpers
    # ------------------------------------------------------------------

    def _most_common(self, counts: Counter) -> Optional[int]:
        return self._get_int(counts.most_common(1)[0][0]) if counts else None

    def _most_common_in_genus(self, counts: Counter, genus_latin: str) -> Optional[int]:
        filtered = Counter({
            sp: n for sp, n in counts.items() if sp.startswith(genus_latin + "_")
        })
        return self._most_common(filtered)

    def _default_species_for_genus(self, genus_latin: str) -> int:
        for val in self.species_dbl.values():
            if val[0].startswith(genus_latin + "_"):
                return val[2]
        return self._get_int("Others")

    def _get_int(self, latin_name: str) -> int:
        return self._latin_to_int[latin_name]

    def _resolve(self, counts: Counter, input_class: int) -> int:
        # Core match logic shared by both predict paths.
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

    # ------------------------------------------------------------------
    # Tile map
    # ------------------------------------------------------------------

    def build_data_map(self, pcd: np.ndarray, crs) -> None:
        # Drop stale map before building — no artifacts from a previous file.
        self._tile_map = None
        self._tile_origin = None
        self._tile_crs = None

        xy = pcd[:, :2]
        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)

        n_x = max(1, math.ceil((x_max - x_min) / self.size_m))
        n_y = max(1, math.ceil((y_max - y_min) / self.size_m))

        tile_map: dict = {}

        for ix in range(n_x):
            for iy in range(n_y):
                cx = x_min + (ix + 0.5) * self.size_m
                cy = y_min + (iy + 0.5) * self.size_m
                lat, lon = self.utm_to_latlon(cx, cy, crs=crs)
                tile_map[(ix, iy)] = self._count_species_in_area(lat, lon)

        self._tile_map = tile_map
        self._tile_origin = (x_min, y_min)
        self._tile_crs = crs

    def _tile_index(self, x: float, y: float) -> tuple:
        ox, oy = self._tile_origin
        ix = int((x - ox) / self.size_m)
        iy = int((y - oy) / self.size_m)
        # clamp — points exactly on the far edge land in the last tile
        ix = min(ix, max(k[0] for k in self._tile_map))
        iy = min(iy, max(k[1] for k in self._tile_map))
        return (ix, iy)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_label(tree_label) -> int:
        # Accepts int, np.integer, single-element ndarray, or single-element list.
        if isinstance(tree_label, (np.ndarray, list)):
            arr = np.asarray(tree_label).ravel()
            if arr.size != 1:
                raise ValueError(f"tree_label must be a scalar, got shape {arr.shape}")
            return int(arr[0])
        return int(tree_label)

    def predict(self, pcd: np.ndarray, crs, tree_label) -> int:
        tree_label = self._coerce_label(tree_label)

        centroid = pcd.mean(axis=0)

        if self._tile_map is not None:
            idx = self._tile_index(centroid[0], centroid[1])
            counts = self._tile_map[idx]
        else:
            lat, lon = self.utm_to_latlon(centroid[0], centroid[1], crs=crs)
            counts = self._count_species_in_area(lat, lon)

        return self._resolve(counts, tree_label)

    def find_species(self, lat: float, lon: float, input_class: int) -> int:
        counts = self._count_species_in_area(lat, lon)
        return self._resolve(counts, input_class)


if __name__ == "__main__":
    bdl = BDLCall(size_m=5000, model_based=True)
    print(bdl.find_species(53.643773, 22.465687, 15))

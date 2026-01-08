import csv


cesta = r"programklo\Rozhledny_v_Jihomoravsk%C3%A9m_kraji___Lookout_towers_in_the_South_Moravian_region.csv"

max_vyska = 0
min_vyska = 8000
max_stavba = 0
min_stavba = 4000


with open(cesta, encoding="utf-8") as file:
    for radek in csv.DictReader(file):
        vyska = int(radek["nadm_vyska"])
        postaveni = int(radek["v_provozu"]) if radek["v_provozu"].isdigit() else postaveni
        max_vyska = max(max_vyska, vyska)
        min_vyska = min(min_vyska,vyska)
        max_stavba = max(max_stavba,postaveni)
        min_stavba = min(min_stavba,postaveni)
        print(f"Rozhledna {radek['nazev']} má nadmořskou výšku {vyska} m n. m.")
    print(f"Nejvyšší rozhledna má {max_vyska}")
    print(f"Nejnižší rozhledna má {min_vyska}")
    print(f"Nejnovější rozhledna postavena v roce {max_stavba}")
    print(f"Nejstarší rozhledna postavena v roce {min_stavba}")
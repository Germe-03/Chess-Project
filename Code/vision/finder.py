"""
Brett in einem Screenshot finden.

Ohne diesen Schritt ist die Erkennung praktisch unbrauchbar: Die Kacheln werden
durch stures Achteln des Bildes gewonnen, und sobald ein Rand, Koordinaten oder
eine Seitenleiste mit auf dem Bild sind, liegen die Kachelgrenzen mitten in den
Feldern. Gemessen fiel die Sicherheit dann von 1.000 auf 0.65, und die erkannte
Stellung war Unsinn.

Das verwendete Merkmal ist nicht die Kantenstruktur, sondern die **Farbe**: Ein
digitales Brett besteht aus genau zwei abwechselnden Flächenfarben und ist
meistens das grösste solche Farbpaar im Bild. Ein erster Versuch über
gleichmässig verteilte Kanten scheiterte, weil Figuren zu viele konkurrierende
Kanten erzeugen.

Ein Farbpaar allein genügt allerdings nicht — ein dunkler Fensterrahmen in
ähnlicher Farbe lässt den Ausschnitt mitwachsen. Deshalb wird jeder Kandidat
zusätzlich darauf geprüft, ob seine 8x8-Unterteilung wirklich abwechselnd hell
und dunkel ist.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

SCHACHBRETT_MUSTER = (np.add.outer(np.arange(8), np.arange(8)) % 2) == 0


def _musterguete(grau: np.ndarray, kasten: tuple[int, int, int, int]) -> float:
    """
    Wie schachbrettartig ist dieser Ausschnitt?

    Gemessen in jeder der 64 Zellen die Helligkeit einer Ecke — nicht der Mitte,
    dort stehen die Figuren. Liegt ein echtes Brett vor, trennen sich die
    Helligkeiten sauber in zwei Gruppen entlang des Schachbrettmusters.
    Rückgabe: Abstand der Gruppen, gemessen in ihrer eigenen Streuung.
    """
    x0, y0, x1, y1 = kasten
    breite, hoehe = (x1 - x0) / 8.0, (y1 - y0) / 8.0
    if breite < 6 or hoehe < 6:
        return 0.0

    rand_x, rand_y = max(1, int(breite * 0.12)), max(1, int(hoehe * 0.12))
    felder = np.zeros((8, 8), dtype=np.float32)
    for reihe in range(8):
        for linie in range(8):
            cx = int(x0 + linie * breite)
            cy = int(y0 + reihe * hoehe)
            ecke = grau[cy + rand_y:cy + int(hoehe * 0.38),
                        cx + rand_x:cx + int(breite * 0.38)]
            felder[reihe, linie] = ecke.mean() if ecke.size else 0.0

    hell = felder[SCHACHBRETT_MUSTER]
    dunkel = felder[~SCHACHBRETT_MUSTER]
    streuung = np.sqrt((hell.var() + dunkel.var()) / 2) + 1e-6
    return float(abs(hell.mean() - dunkel.mean()) / streuung)


def find_board(image: Image.Image, max_width: int = 480, min_pattern: float = 2.0):
    """
    Ausschnitt (links, oben, rechts, unten) des Bretts, oder None.

    None heisst: kein Brett sicher erkannt. Die Oberfläche soll dann zum
    Zuschneiden von Hand auffordern, statt still ein falsches Ergebnis zu
    liefern.
    """
    bild = image.convert("RGB")
    faktor = max(1.0, max(bild.size) / max_width)
    klein = bild.resize((max(1, int(bild.width / faktor)), max(1, int(bild.height / faktor))), Image.NEAREST)
    farben = np.asarray(klein, dtype=np.uint8)
    grau = np.asarray(klein.convert("L"), dtype=np.float32)

    # Farben grob zusammenfassen, damit Kompressionsartefakte nicht stören
    grob = (farben >> 3).astype(np.int32)
    schluessel = grob[:, :, 0] * 1024 + grob[:, :, 1] * 32 + grob[:, :, 2]
    werte, anzahl = np.unique(schluessel, return_counts=True)
    haeufig = werte[np.argsort(-anzahl)][:8]

    bestes = (0.0, None)
    for i in range(len(haeufig)):
        for j in range(i + 1, len(haeufig)):
            maske = (schluessel == haeufig[i]) | (schluessel == haeufig[j])
            if maske.sum() < 400:
                continue
            ys, xs = np.nonzero(maske)
            kasten = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            breite, hoehe = kasten[2] - kasten[0], kasten[3] - kasten[1]
            if breite < 40 or hoehe < 40:
                continue
            seitenverhaeltnis = min(breite, hoehe) / max(breite, hoehe)
            if seitenverhaeltnis < 0.9:
                continue
            fuellung = maske[kasten[1]:kasten[3], kasten[0]:kasten[2]].sum() / (breite * hoehe)
            if fuellung < 0.45:
                continue
            muster = _musterguete(grau, kasten)
            if muster < min_pattern:
                continue
            punkte = muster * fuellung * breite * hoehe
            if punkte > bestes[0]:
                bestes = (punkte, kasten)

    if bestes[1] is None:
        return None
    return tuple(int(round(v * faktor)) for v in bestes[1])

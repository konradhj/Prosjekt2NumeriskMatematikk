Vi simulerte begge modellene med $T=10^5$ og steglengde $h=0.1$. Referanseløsningen ble beregnet med `solve_ivp` og DOP853, med `rtol=1e-12` og `atol=1e-14`.

Alle tre metodene gjengir hovedformen til partikkelbanene. I 2D-modellen ser vi raske rotasjoner kombinert med en langsom drift rundt origo. For tokamak-modellen viser plottene i $(R,x_3)$-planet de forventede bananbanene. Banene ser svært like ut mellom metodene, men dette betyr ikke at posisjonene er like ved samme tidspunkt. Bananplottet skjuler dessuten vinkelen rundt tokamakens akse, slik at enkelte posisjonsforskjeller ikke er synlige i denne projeksjonen.

Posisjonsfeilen øker over tid, med oscillasjoner og perioder der feilen avtar. Dette er forenlig med faseforskjeller: Partiklene kan følge omtrent samme bane, men befinne seg på forskjellige steder langs den. I langtidsforsøket får RK4 størst posisjonsfeil ved sluttiden for begge modellene. Lignende baneplott er derfor ikke tilstrekkelig til å vurdere metodens nøyaktighet.

Energiplottene viser en tydelig forskjell mellom metodene. RK4 får en gradvis voksende energifeil, mens Boris og den symmetriske flerstegsmetoden holder energifeilen omtrent innenfor faste grenser gjennom intervallet. Den symmetriske metoden har minst maksimal energifeil i begge modellene. Dette samsvarer med forventningen om god langtidsbevaring av energi. De raske energiutsvingene er ikke vedvarende drift; blokkmaksimaene gjør denne forskjellen lettere å se.

For tokamak-modellen har Boris et lite, tilnærmet konstant energiavvik. Her beregnes heltallshastigheten som gjennomsnittet av to halvstegshastigheter. Denne middelverdien kan gi et energiavvik selv om normen til halvstegshastighetene bevares når det elektriske potensialet er null.

Resultatene viser dermed den forventede forskjellen mellom god nøyaktighet over kortere intervaller og god bevaring av energi over lang tid. God energibevaring alene garanterer likevel ikke liten posisjonsfeil.
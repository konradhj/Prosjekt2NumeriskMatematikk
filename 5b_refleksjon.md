### Ordens verifikasjon

For den todimensjonale ladede partikkelen observeres et lite avvik mellom RK4-kurven og den teoretiske h4h^4h4-referansen. Avviket er imidlertid lite, og siden analysen kun er basert på tre steglengder og det brukes en numerisk referanseløsning, vurderes fortsatt resultatene som konsistente med den teoretiske fjerdeordens konvergensen. Boris-metoden og den en symmetriske multistep-metoden følger referansene sine svært godt, henholdsvis h^2 og h^4

For tokamak-systeme stemmer de observerte feilene og de teoretiske referanselinjene enda bedre. Her er det ingen synloge avvik mellom teoretisk  og observert feil. 

Totalt sett viser resultatene god overensstemmelse med teorien, og de forventede konvergensordnene ser ut til å være oppnådd for alle tre metodene.

### Wrok precision

Work-precision-diagrammet viser en tilnærmet lineær sammenheng mellom feil og beregningstid for RK4 og den symmetriske multistep-metoden, med ganske likt stigningstall. Boris-metoden ligger betydelig lenger til høyre, uten å ligge særlig lavere, enn de to andre metodene og oppnår dermed større feil for de undersøkte steglengdene. RK4 oppnår de laveste feilene, mens multistep-metoden oppnår noe høyere feil til en lavere beregningskostnad.


De samme tendensene observeres for tokamak-systemet. RK4 gir de laveste feilene, mens multistep-metoden oppnår sammenlignbare feilnivåer ved lavere beregningskostnad. Boris-metoden ligger igjen betydelig lenger til høyre i diagrammet enn de to fjerdeordens metodene.


Resultatene viser at Boris-metoden er minst effektiv av de tre metodene for de undersøkte steglengdene. Den er både tregere og gir mindre nøyaktige resultater enn multistep-metoden. RK4 gir generelt de laveste feilene, men til en høyere beregningskostnad. Multistep-metoden følger en svært lignende trend og oppnår noe høyere feil, men til en lavere beregningskostnad. Ved sammenlignbare feilnivåer, slik som mellom RK4 med den minste steglengden og multistep-metoden med den største steglengden, fremstår multistep-metoden som noe mer effektiv.
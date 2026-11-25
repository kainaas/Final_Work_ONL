# Plano: transições e cenas extras do vídeo

Este é só o plano: nada daqui foi implementado nem renderizado ainda.

## O tamanho do problema

| | Tempo |
|---|---|
| Meta do vídeo | 10:00 |
| Narração atual de `roteiro.md` (960 palavras a ~140 por minuto) | ~6:50 |
| Animação já renderizada (11 cenas) | 2:31 |

Faltam duas coisas, não uma:

1. **Imagem:** a animação cobre só um quarto do vídeo.
2. **Fala:** o roteiro dá menos de 7 minutos falando com calma, não 10. A
   estimativa de 1.400 palavras que escrevi no roteiro estava errada.

A proposta fecha os 10 minutos assim: **8:30 de narração** (o roteiro ganha cerca
de 250 palavras em três pontos, listados abaixo) e **1:30 só de imagem**
(cartões de seção, pausas e a vinheta). A animação passa de 2:31 para cerca de 8:30.
O resto fica para filmagens opcionais (torneira, bolha, inseto).

## Linha do tempo proposta

| Seção | Narração | Animação hoje | Animação proposta | Como fechar a diferença |
|---|---|---|---|---|
| 0. Vinheta | — | 0 | 0:06 | `Vinheta` (nova) |
| 1. Abertura | 0:32 | 0:10 | 0:40 | `Abertura` com a gota pingando em loop (+0:30) |
| 2. A pele da água | 0:47 → 1:10 | 0:08 | 1:05 | `CartaoSecao` + `Moleculas` (nova) + `PeleDaAgua` mais lenta |
| 3. A gravidade puxa | 0:34 | 0:16 | 0:40 | `CartaoSecao` + `CaboDeGuerra` com pausas |
| 4. Otimizar | 1:11 → 1:35 | 0:24 | 1:35 | `CartaoSecao` + `EnergiaViraMorro` (transição) + `ModeloQuadratico` (nova) |
| 5. O computador | 1:00 | 0:16 | 1:05 | `CartaoSecao` + `TrilhaViraGota` (transição) + `RaiosDosPontos` (nova) + `GotaSeAjeitando` mais lenta |
| 6. Crescendo até cair | 0:48 | 0:16 | 0:55 | `CartaoSecao` + `Torneira` (transição) + `CurvaVolumePressao` (nova) |
| 7. Em 3D | 1:25 → 1:55 | 0:47 | 1:55 | `CartaoSecao` + `GiroVira3D` (transição) + `ForcaDaParede` (nova) |
| 8. Para que serve | 0:35 | 0:13 | 0:40 | `ZoomOut` (transição) + créditos |
| **Total** | **~8:30** | **2:31** | **~8:40** | |

## Transições (ligam uma seção à outra)

São curtas, de 4 a 15 s, e a ideia é que a última imagem de uma cena *vire* a
primeira da seguinte, para o espectador não sentir um corte.

1. **`Vinheta`** (6 s): uma gota cai sobre o título "A forma de uma gota" e as
   letras ondulam.
2. **`CartaoSecao`** (4 s cada, 7 cartões = 28 s): número e título da seção, com uma
   gotinha que escorre da esquerda para a direita. Uma única classe com parâmetro
   gera os 7 vídeos.
3. **`EnergiaViraMorro`** (3 → 4, 8 s): a curva "energia total × quanto a gota
   está esticada" do `CaboDeGuerra` se solta dos eixos e vira o morro do
   `DescendoOMorro`. O ponto de equilíbrio vira a bolinha no fundo do vale.
4. **`TrilhaViraGota`** (4 → 5, 10 s): a elipse da `TrilhaRestrita` se deforma no
   perfil da gota, e a bolinha se divide nos 30 pontos do perfil.
5. **`Torneira`** (5 → 6, 6 s): uma torneira estilizada aparece sobre o bico e o
   contador de volume começa a subir.
6. **`GiroVira3D`** (6 → 7, 15 s): o perfil 2D gira em torno do eixo e desenha a
   superfície de revolução, que depois se quebra nos triângulos da malha. É a
   transição mais importante: mostra por que o caso 2D já era uma gota 3D
   redonda.
7. **`ZoomOut`** (7 → 8, 8 s): a câmera se afasta da gota 3D até ela virar um
   ícone pequeno ao lado dos três ícones das aplicações.

## Cenas novas (preenchem o meio das seções)

1. **`Moleculas`** (seção 2, 20 s): bolinhas representando moléculas de água. No
   meio do líquido, cada uma é puxada igualmente para todos os lados; na
   superfície, só para dentro. Setas mostram o puxão, e daí vem a "pele".
2. **`ModeloQuadratico`** (seção 4, 25 s): na `TrilhaRestrita`, a cada passo
   aparece em volta do ponto atual o "mapa simplificado" (curvas de nível do
   modelo quadrático, em elipses) e a reta tangente à trilha. O próximo ponto é o
   mínimo desse mapa sobre a reta. É o SLSQP visto de perto, sem fórmula.
3. **`RaiosDosPontos`** (seção 5, 15 s): os raios que saem do centro da calota e
   cada ponto do perfil deslizando sobre o seu raio. Mostra o que é "cada ponto só
   pode andar para dentro ou para fora".
4. **`CurvaVolumePressao`** (seção 6, 20 s): o gráfico de volume × pressão no
   ápice (os dados de `curva_VA.pdf`), com um ponto andando sobre a curva enquanto a
   gota cresce ao lado. No volume máximo a curva dá meia-volta, e fica claro que
   não existe gota maior.
5. **`ForcaDaParede`** (seção 7, 20 s): a gota com a parede, com os 20 vértices
   que encostam nela em laranja e setas proporcionais aos multiplicadores
   $\sigma_j$ (a "força" da parede). Liga o vídeo à Tabela 7 do relatório.

## Cenas existentes que ficam mais lentas

Sem mudar o conteúdo, só os tempos (`run_time` e `wait`):

| Cena | Hoje | Proposta | O que muda |
|---|---|---|---|
| `Abertura` | 0:10 | 0:40 | a gota cresce, se solta e recomeça (2 ciclos) |
| `PeleDaAgua` | 0:08 | 0:25 | uma figura por vez, com o contador da borda subindo |
| `CaboDeGuerra` | 0:16 | 0:30 | vai e volta duas vezes antes de parar no equilíbrio |
| `GotaSeAjeitando` | 0:16 | 0:40 | 1 s por passo e pausa na comparação com a EDO |
| `CrescendoAteCair` | 0:16 | 0:30 | pausa a cada 5 de volume |
| `Gota3D` | 0:19 | 0:35 | volta completa da câmera antes e depois |
| `QuadradoEParede` | 0:21 | 0:30 | pausa em cada gota |

## Narração a acrescentar (cerca de 250 palavras)

1. **Seção 2:** um parágrafo sobre as moléculas (acompanha `Moleculas`).
2. **Seção 4:** explicar o "mapa simplificado" passo a passo (acompanha
   `ModeloQuadratico`).
3. **Seção 7:** a parede "empurra" a gota, e o computador calcula quanto ela
   empurra em cada ponto (acompanha `ForcaDaParede`).

Se você preferir não aumentar a fala, dá para gravar mais devagar (~115 palavras
por minuto chega a ~8:20) e deixar mais tempo para as pausas só com imagem.

## Ordem de trabalho sugerida

1. Escrever os trechos novos do roteiro (você revisa o texto antes).
2. Gravar a narração. Com o áudio pronto, os tempos de cada cena saem do áudio
   e não de estimativa.
3. Implementar as transições e cenas novas, ajustando `run_time` ao áudio.
4. Montar no editor (Kdenlive, Shotcut ou DaVinci Resolve): cada cena já sai
   como um `.mp4` separado.

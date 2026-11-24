# Plano de pesquisa — EFNNE 2026

## Objetivo geral
Implementar e avaliar PINNs para aproximar soluções de equações diferenciais parciais,
quantificando erro, convergência e custo computacional em problemas com referência conhecida.

## Objetivos específicos
1. Implementar PINN para a equação 1D do calor.
2. Comparar a predição com a solução analítica em pontos de teste independentes.
3. Implementar uma formulação real/imag. para a equação de Schrödinger dependente do tempo,
   usando uma onda plana periódica como referência analítica compatível.
4. Avaliar erro, perda de treinamento, tempo de treinamento e densidade de probabilidade.
5. Documentar limitações e condições sob as quais os resultados são reproduzíveis.

## Hipóteses a testar (não são resultados)
- H1: o PINN pode aproximar a solução do calor com erro mensurável e controlável.
- H2: o equilíbrio entre resíduo da EDP e condições iniciais/de contorno influencia a precisão.
- H3: o desempenho do PINN depende da amostragem, arquitetura e otimização.
- H4: a formulação de Schrödinger exige atenção adicional à parte complexa e à compatibilidade
  entre domínio, condições de contorno e solução de referência.

## Variáveis e métricas
- Erro L2 relativo: ||u_pred-u_ref||_2 / ||u_ref||_2.
- RMSE: sqrt(mean((u_pred-u_ref)^2)); para Schrödinger, usar módulo complexo.
- Erro absoluto máximo.
- Tempo de treinamento (segundos).
- Perdas separadas: EDP, condição inicial e contorno.
- Schrödinger: densidade |psi|^2 e integral espacial de |psi|^2 em t=0.

## Controles experimentais necessários
- Fixar semente aleatória e registrar dispositivo/versões.
- Separar pontos de colocação e pontos de avaliação.
- Fazer pelo menos 3 sementes por configuração antes de reportar médias/desvios.
- Comparar configurações com o mesmo orçamento computacional quando possível.
- Registrar falhas de convergência.
- Não afirmar superioridade sobre métodos numéricos sem uma comparação justa.

## Alerta metodológico importante
O experimento principal em `src/schrodinger_pinn.py` usa uma onda plana livre, periódica e
compatível com o domínio e as condições de contorno. Isso fornece uma referência analítica
exata, mas não testa um pacote localizado. A implementação gaussiana exploratória permanece
incompatível com o domínio periódico e não deve ser apresentada como resultado validado.
Antes de tirar conclusões gerais, executar múltiplas sementes por configuração, verificar a
conservação da norma ao longo do tempo e, se o objetivo incluir pacotes localizados, definir
um problema de referência e condições de contorno compatíveis.

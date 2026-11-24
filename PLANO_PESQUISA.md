# Plano de pesquisa — EFNNE 2026

## Objetivo geral
Implementar e avaliar PINNs para aproximar soluções de equações diferenciais parciais,
quantificando erro, convergência e custo computacional em problemas com referência conhecida.

## Objetivos específicos
1. Implementar PINN para a equação 1D do calor.
2. Comparar a predição com a solução analítica em pontos de teste independentes.
3. Implementar uma formulação real/imag. para a equação de Schrödinger dependente do tempo.
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
- Schrödinger: densidade |psi|^2 e integral espacial de |psi|^2.

## Controles experimentais necessários
- Fixar semente aleatória e registrar dispositivo/versões.
- Separar pontos de colocação e pontos de avaliação.
- Fazer pelo menos 3 sementes por configuração antes de reportar médias/desvios.
- Comparar configurações com o mesmo orçamento computacional quando possível.
- Registrar falhas de convergência.
- Não afirmar superioridade sobre métodos numéricos sem uma comparação justa.

## Alerta metodológico importante
A referência gaussiana de Schrödinger no script inicial é a solução livre em domínio infinito,
enquanto o PINN recebe condições periódicas em domínio finito. Isso introduz incompatibilidade.
Antes de publicar resultados de Schrödinger, substituir por um caso compatível (por exemplo,
potencial harmônico com referência numérica de alta resolução e condições de contorno justificadas,
ou uma solução periódica exata) e validar a formulação.

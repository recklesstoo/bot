# Comentarios sobre TraderOS (versión corta)

Son impresiones de una sesión de investigación por MCP. Varias pueden ser decisiones de diseño; van
como preguntas o sugerencias, no como errores. La evidencia detallada está en `traderos_feedback.md`.

## Lo que funcionó bien
- `evaluate_edge` dio exactamente lo mismo que mi cálculo propio (178 operaciones, t = -0,498).
- Las herramientas de resultados y validación responden en menos de medio segundo.
- Etiquetas como "INSUFFICIENT SAMPLE" y "needs more evidence" son honestas y útiles.

## Comentarios pequeños
1. **`quant_discovery_scan`:** las dos veces que lo probé terminó en `Could not reach TraderOS API ... .`
   sin causa. Puede ser transitorio, pero ¿se podría mostrar el motivo?
2. **Walk-forward:** en un backtest de 2021 a 2026, los pliegues empezaban en 2024. ¿Es intencional? Si lo
   es, ayudaría decirlo en el resultado.
3. **Cifras de confianza:** una misma estrategia mostraba 46%, 76,58% y "robust". ¿Se podría indicar
   qué mide cada una? Con 9 operaciones, quizá convenga no mostrar confianza.
4. **Señales no reconciliadas:** "140 señales no reconciliadas" frente a 9 entradas. ¿Se podría dar el
   motivo por señal? Puede ser lo esperado por los filtros.
5. **Datos de ES:** el volumen cae en las semanas de vencimiento trimestral. ¿Se cambia de contrato por
   volumen? Documentar la regla ayudaría a quien use volumen relativo.
6. **Consultas en UTC:** un parámetro de sesión (por ejemplo, 09:30-11:00 hora de Nueva York) evitaría
   errores con el cambio de hora.
7. **Tamaño de las respuestas:** algunas repiten filas o argumentos varias veces (`query_data`,
   `evaluate_edge`), y eso consume mucho contexto.
8. **Catálogo de filtros:** me habrían servido ventanas de 30 minutos, bandas de VWAP en desviaciones
   estándar y volumen relativo por vela.
9. **Conexión de solo lectura:** entiendo que es por seguridad. Un mensaje más claro al rechazar una
   acción, con el motivo, ayudaría.

window.MathJax = {
    tex2jax: {
      inlineMath: [['$', '$'], ["\\(", "\\)"]],
      displayMath: [['$$', '$$'], ["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true,
      ignoreClass: ".*|",
      processClass: "arithmatex"
    },
    TeX: {
      TagSide: "right",
      TagIndent: ".8em",
      MultLineWidth: "85%",
      equationNumbers: {
        autoNumber: "AMS",
      },
      unicode: {
        fonts: "STIXGeneral,'Arial Unicode MS'"
      }
    },
    showProcessingMessages: false,
    messageStyle: "none",
    jax: ["input/TeX","output/SVG"]
};
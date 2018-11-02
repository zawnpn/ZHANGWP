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
(function(u, c) {
  var d = document, t = 'script', o = d.createElement(t),
      s = d.getElementsByTagName(t)[0];
  o.src = u;
  if (c) { o.addEventListener('load', function(e) { c(e); }); }
  s.parentNode.insertBefore(o, s);
})('//cdn.bootcss.com/pangu/3.3.0/pangu.min.js', function() {
  pangu.spacingPage();
});
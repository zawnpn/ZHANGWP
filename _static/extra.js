//pangu
document.addEventListener('DOMContentLoaded', () => {
// listen to any DOM change and automatically perform spacing via MutationObserver()
  pangu.autoSpacingPage();
});
//mathjax
window.MathJax = {
    tex: {
      inlineMath: [ ["\\(","\\)"] ],
      displayMath: [ ["\\[","\\]"] ],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    }
  };
// // Re-render MathJax on document switch (instant loading, custom event)
// document.addEventListener("DOMContentSwitch", function() {
//   MathJax.typesetPromise();
// });
var _hmt = _hmt || [];
(function() {
  var hm = document.createElement("script");
  hm.src = "https://hm.baidu.com/hm.js?d919f3c6882717ddb93c07ee58cb658e";
  var s = document.getElementsByTagName("script")[0]; 
  s.parentNode.insertBefore(hm, s);
})();
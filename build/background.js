(() => {
  "use strict";
  let e = null,
    t = null,
    n = null;
  const s = new ArrayBuffer(1),
    o = new DataView(s);
  chrome.runtime.onMessage.addListener((t, a, c) => {
    switch (t.type) {
      case "RES":
        e?.send(t.payload);
        break;
      case "END":
        o.setUint8(0, 0), e?.send(s);
        break;
      case "ERROR":
        o.setUint8(0, 1), e?.send(s);
        break;
      case "URL":
        chrome.tabs.query(
          { url: chrome.runtime.getManifest().content_scripts[0].matches[0] },
          function (e) {
            e.length &&
              (chrome.tabs.update(e[0].id, {
                url: "https://poe.com/" + t.payload.model,
              }),
              (n = () => {
                chrome.tabs.query(
                  {
                    url: chrome.runtime.getManifest().content_scripts[0]
                      .matches[0],
                  },
                  function (e) {
                    e.length &&
                      chrome.tabs.sendMessage(
                        e[0].id,
                        { type: "RESEND", payload: t.payload },
                        function (e) {}
                      );
                  }
                );
              }));
          }
        );
      case "READY":
        n && n(), (n = null);
    }
    return c({}), !0;
  }),
    (function n() {
      (e = new WebSocket("ws://127.0.0.1:18765/ws")),
        (e.onopen = (n) => {
          console.log("websocket open"),
            (function () {
              const t = setInterval(() => {
                e ? (o.setUint8(0, 255), e.send(s)) : clearInterval(t);
              }, 2e4);
            })(),
            t && (clearInterval(t), (t = null));
        }),
        (e.onmessage = (e) => {
          console.log(`websocket received message: ${e.data}`),
            chrome.tabs.query(
              {
                url: chrome.runtime.getManifest().content_scripts[0].matches[0],
              },
              function (t) {
                for (let n of t) {
                  chrome.tabs.sendMessage(
                    n.id,
                    {
                      type: "SEND",
                      payload: JSON.parse(
                        e.data.replace(/\n/g, "\\n").replace(/\r/g, "\\r")
                      ),
                    },
                    function (e) {}
                  );
                  break;
                }
              }
            );
        }),
        (e.onclose = (s) => {
          console.log("websocket connection closed"),
            (e = null),
            t || (t = setInterval(n, 5e3));
        });
    })();
})();
//# sourceMappingURL=background.js.map

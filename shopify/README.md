# App Shopify «Prova nella tua stanza»

Questa cartella è l'app Shopify da collegare al servizio `posa server`.

## Cosa fa
- aggiunge alla pagina prodotto un blocco (Theme App Extension) con il pulsante
  "Prova nella tua stanza": il cliente carica la foto e vede il rivestimento posato;
- tramite l'**App Proxy** (`https://negozio/apps/posa/...`) il negozio inoltra le richieste
  al servizio firmandole: il servizio sa con certezza da quale negozio arrivano;
- riceve i **webhook** dei prodotti (catalogo sempre aggiornato) e quelli **GDPR**.

## Installazione (una volta, per l'app)
1. Shopify Partner Dashboard → Apps → Create app → "Create app manually".
2. Copia il **Client ID** in `shopify.app.toml` e sostituisci `SERVER` con l'indirizzo
   pubblico del servizio.
3. Sul server imposta `SHOPIFY_API_SECRET` con il **Client secret** dell'app.
4. Con la Shopify CLI: `shopify app deploy` (pubblica configurazione ed estensione).

## Per ogni negozio cliente
1. Registra il negozio sul servizio (token di amministrazione):
   `curl -X POST https://SERVER/admin/negozi -H "Authorization: Bearer $POSA_ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"dominio":"negozio.myshopify.com","sito":"https://www.negozio.it","prove_al_giorno":100}'`
2. Prima sincronizzazione del catalogo:
   `curl -X POST https://SERVER/admin/negozi/negozio.myshopify.com/sincronizza -H "Authorization: Bearer $POSA_ADMIN_TOKEN"`
3. Il negozio installa l'app e, nell'editor del tema, aggiunge il blocco
   "Prova nella tua stanza" al modello prodotto.

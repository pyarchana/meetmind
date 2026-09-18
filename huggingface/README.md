# Deploying to Hugging Face Spaces

Notes for putting MeetMind back online for free. Nothing here runs
automatically until you create the Space and add a token.

## Why Spaces

The app needs long lived WebSockets, a Python ASGI server, and no money. That
combination rules out most free tiers. Spaces gives a free CPU container with
no card, no cap on connection duration, a secrets UI, and Docker support so
the existing `app/Dockerfile` works unchanged. It sleeps after inactivity and
wakes on the next visit.

Vercel now supports FastAPI WebSockets but closes connections at 300 seconds
on Hobby, and a reconnect can land on a different instance where
`InMemorySessionService` has never heard of the session, so the meeting board
would empty every five minutes. That needs persistent sessions first.

## One time setup

**1. Create the Space.** At [huggingface.co/new-space](https://huggingface.co/new-space):
name it `meetmind`, pick **Docker** as the SDK and **Blank** as the template,
and leave hardware on the free CPU tier.

**2. Add the key.** In the Space, Settings then Variables and secrets, add a
secret named `GOOGLE_API_KEY`. Use a fresh key, not one of the three that
leaked in March.

**3. Push the app.** Either let the GitHub Action do it, or once by hand:

```bash
git clone https://huggingface.co/spaces/<your-username>/meetmind hf-space
cp -r app/. hf-space/
rm -rf hf-space/tests hf-space/pytest.ini hf-space/requirements-dev.txt
cp huggingface/space-readme.md hf-space/README.md
cd hf-space && git add -A && git commit -m "deploy meetmind" && git push
```

The Space root ends up as the contents of `app/`, so `app/Dockerfile` sits at
the root where Spaces expects it. `app_port: 8080` in the Space README tells
Spaces where to route, matching the `PORT` default in the Dockerfile.

## Automatic deploys

`.github/workflows/deploy-space.yml` pushes on every commit to main, and does
nothing at all until both of these exist:

- repository **secret** `HF_TOKEN`, a write token from
  [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- repository **variable** `HF_SPACE`, set to `<your-username>/meetmind`

Until then the workflow logs that it is skipping and exits green, so it will
not sit red in your CI history while you decide.

## Things worth knowing before you share the link

**The endpoint is open.** Anyone with the URL can start a session against your
key. On the free tier that means rate limits rather than a bill, but a busy
day makes the demo look broken. The same concern that has kept
`--allow-unauthenticated` on the open items list since the first review.

**First visit after a sleep is slow.** The container has to wake, which takes
tens of seconds. Worth saying so on the page if you are sending the link to
someone who will judge it.

**The build is about a gigabyte.** `google-adk` pulls a lot. Spaces handles it,
but the first build is not quick.

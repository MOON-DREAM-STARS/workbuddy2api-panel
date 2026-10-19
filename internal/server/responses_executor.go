package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
	"github.com/linguo2625469/workbuddy2api-panel/internal/logfmt"
	"github.com/linguo2625469/workbuddy2api-panel/internal/pool"
	"github.com/linguo2625469/workbuddy2api-panel/internal/prompt"
	"github.com/linguo2625469/workbuddy2api-panel/internal/reqlog"
	"github.com/linguo2625469/workbuddy2api-panel/internal/session"
	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
	"github.com/linguo2625469/workbuddy2api-panel/internal/usage"
)

type ChatExecutionRequest struct {
	Context                    context.Context
	Body                       []byte
	Model                      string
	Stream                     bool
	Headers                    http.Header
	ClientIP                   string
	ResponsesDiagnostics       bool
	ResponsesCandidateSections []string
}

type ChatRenderMeta struct {
	Account     *auth.Auth
	BareModel   string
	Realm       string
	ReqHasImage bool
	ChatMeta    upstream.ChatMeta
	Started     time.Time
	PriorUsage  map[string]any
}

type ChatOutcome struct {
	Status          int
	FinishReason    *string
	SawDone         bool
	ReadErr         error
	UpstreamErr     error
	RendererErr     error
	OutputCommitted bool
	Response        any

	TTFB       time.Duration
	Tokens     int
	HasUsage   bool
	Prompt     int
	CacheHit   int
	CacheMiss  int
	CacheWrite int
	Credit     float64
	HasCredit  bool
}

type ChatRenderer interface {
	RenderStream(context.Context, io.ReadCloser, ChatRenderMeta) ChatOutcome
	RenderNonStream(context.Context, io.ReadCloser, ChatRenderMeta) ChatOutcome
}

type chatFailure struct {
	Status  int
	Code    string
	Message string
	Hint    string
	Param   string
}

// executeChat is the single owner of realm routing, prompt policy, sticky
// routing, rotation, cooldowns, leases, cancellation, cost and request metrics.
// The renderer finishes before this function releases the account lease.
func (h *Handler) executeChat(req ChatExecutionRequest, renderer ChatRenderer) (ChatOutcome, *chatFailure) {
	ctx := req.Context
	if ctx == nil {
		ctx = context.Background()
	}
	var peek struct {
		Stream bool   `json:"stream"`
		Model  string `json:"model"`
	}
	_ = json.Unmarshal(req.Body, &peek)
	if req.Model == "" {
		req.Model = peek.Model
	}
	if !req.Stream {
		req.Stream = peek.Stream
	}

	realm, bareModel := resolveModel(req.Model)
	emptyToolRetryEnabled := h.emptyToolTurnRetryEnabled(req.Model, req.Body)
	st := newChatStat(time.Now(), req.Body, req.Stream)
	if trace := requestTraceFromContext(ctx); trace != nil {
		trace.stat = st
	}
	defer st.done()

	tried := map[string]bool{}
	var lastErr error

	sessKey := session.ExtractKey(req.Body)
	stickyKey := sessKey
	stickyUID := ""
	if h.cfg.Session != nil && stickyKey != "" {
		if uid, ok := h.cfg.Session.ResolveForModel(stickyKey, req.Model); ok {
			stickyUID = uid
		}
	}
	turnKey := session.TurnKey(req.Body)
	reqHasImage := hasImagePart(req.Body)

	var heldUID string
	defer func() {
		if heldUID != "" {
			h.cfg.Pool.Release(heldUID)
		}
	}()
	releaseHeld := func() {
		if heldUID != "" {
			h.cfg.Pool.Release(heldUID)
			heldUID = ""
		}
	}
	unbindSticky := func() {
		if stickyUID != "" && h.cfg.Session != nil {
			h.cfg.Session.Unbind(stickyKey)
			stickyUID = ""
		}
	}
	fail := func(uid string) {
		releaseHeld()
		if stickyUID != "" && uid == stickyUID {
			unbindSticky()
		}
	}

	degradedApplied := false
	if h.cfg.PromptMode == "custom" && h.cfg.PromptText != "" {
		req.Body = prompt.Rewrite(req.Body, h.cfg.PromptText)
	} else if h.cfg.PromptMode == "append" && h.cfg.PromptText != "" && !h.degrade.Active() {
		req.Body = prompt.Append(req.Body, h.cfg.PromptText)
	} else if (h.cfg.PromptMode == "passthrough" || h.cfg.PromptMode == "append") && h.degrade.Active() {
		req.Body = prompt.Rewrite(req.Body, prompt.Degraded)
		degradedApplied = true
	}
	if bareModel != req.Model {
		req.Body = rewriteModel(req.Body, bareModel)
	}

	chatMeta := upstream.ChatMeta{ConversationID: session.ResolveConversationID(req.Body)}
	if v := req.Headers.Get("X-Conversation-Request-ID"); v != "" {
		chatMeta.ConversationRequestID = v
	} else if turnKey != "" && sessKey != "" {
		chatMeta.ConversationRequestID = session.TurnRequestID(sessKey + ":" + turnKey)
	} else if turnKey != "" {
		chatMeta.ConversationRequestID = session.TurnRequestID(turnKey)
	} else if sessKey != "" {
		chatMeta.ConversationRequestID = session.RequestIDForKey(sessKey)
	} else {
		chatMeta.ConversationRequestID = session.TurnRequestID("")
	}
	chatMeta.TraceID = req.Headers.Get("X-Trace-ID")

	emptyRetryState := h.newEmptyToolTurnRetryState(st.start)
	accountAttempts := h.cfg.MaxRotate
	if emptyToolRetryEnabled && emptyRetryState.maxRetries+1 > accountAttempts {
		accountAttempts = emptyRetryState.maxRetries + 1
	}
	for i := 0; i < accountAttempts; i++ {
		var acct *auth.Auth
		if stickyUID != "" {
			acct = h.cfg.Pool.PickByUIDForModel(stickyUID, bareModel)
			if acct == nil || (realm != "" && acct.Realm() != realm) {
				unbindSticky()
			}
		}
		if acct == nil {
			acct = h.cfg.Pool.PickExcludingForRealm(tried, bareModel, realm)
		}
		if acct == nil {
			st.status = http.StatusServiceUnavailable
			break
		}
		st.uid = acct.UID
		st.nick = acct.Nickname
		tried[acct.UID] = true

		if !h.cfg.Pool.Acquire(acct.UID) {
			if stickyUID != "" && acct.UID == stickyUID {
				unbindSticky()
			}
			if !rotateBackoff(i, ctx) {
				break
			}
			continue
		}
		heldUID = acct.UID
		attemptStarted := time.Now()

		if acct.NeedsRefresh(h.cfg.RefreshSkew) {
			if err := h.cfg.Upstream.RefreshToken(acct); err != nil {
				lastErr = err
				var ue *upstream.Error
				if errors.As(err, &ue) && ue.Kind == upstream.ErrSessionDead {
					h.cfg.Pool.NoteSessionDead(acct.UID)
				} else {
					h.cfg.Pool.NoteError(acct.UID)
				}
				fail(acct.UID)
				if !rotateBackoff(i, ctx) {
					break
				}
				continue
			}
			acct.BackfillRealm()
			if err := acct.SaveAtomic(); err != nil {
				log.Printf("ERR: [server] chat refresh acct=%s: save auth failed: %v", logfmt.Label(acct.UID, acct.Nickname), err)
			}
		}

		preparedBody := []byte(nil)
		if emptyToolRetryEnabled {
			preparedBody = h.cfg.Upstream.PrepareChatBody(acct, req.Body, chatMeta)
		}
		var bufferedStats *chatStatsReader
		var finalObservation *upstream.EmptyToolTurnObservation
		var rc io.ReadCloser
		var status int
		var respBody []byte
		var terr error
		retryNextAccount := false
		for {
			attemptStarted = time.Now()
			attemptCtx := ctx
			attemptCancel := func() {}
			localDeadline := false
			if emptyToolRetryEnabled {
				attemptCtx, attemptCancel, localDeadline = emptyRetryState.attemptContext(attemptCtx, attemptStarted)
				rc, status, respBody, terr = h.cfg.Upstream.ChatStreamPreparedContext(attemptCtx, acct, preparedBody, req.ClientIP, chatMeta)
			} else {
				rc, status, respBody, terr = h.cfg.Upstream.ChatStreamContext(ctx, acct, req.Body, req.ClientIP, chatMeta)
			}
			if terr != nil && emptyRetryState.deadlineHit(localDeadline, attemptCtx) {
				attemptCancel()
				recordResponsesUsage(h, st, acct.UID, bareModel, pool.TokenUsageDelta{}, 0, false, attemptStarted)
				recordEmptyToolTurnAttempt(ctx, bareModel, logfmt.Label(acct.UID, acct.Nickname),
					emptyRetryState.retries+1, "deadline", nil, time.Since(attemptStarted), 0)
				st.status, st.outcome = http.StatusGatewayTimeout, reqlog.OutcomeHTTPError
				releaseHeld()
				return ChatOutcome{Status: http.StatusGatewayTimeout}, &chatFailure{
					Status: http.StatusGatewayTimeout, Code: "upstream_empty_tool_turn",
					Message: "empty tool-turn retry budget expired",
				}
			}
			if terr != nil || status >= 400 || !emptyToolRetryEnabled {
				attemptCancel()
				break
			}
			bufferedStats = newChatStatsReaderSince(rc, attemptStarted)
			raw, readErr := readValidatedResponse(bufferedStats)
			_ = rc.Close()
			attemptCancel()
			if readErr != nil && emptyRetryState.deadlineHit(localDeadline, attemptCtx) {
				credit, hasCredit := bufferedStats.Credit()
				recordResponsesUsage(h, st, acct.UID, bareModel, bufferedStats.Usage(), credit, hasCredit, attemptStarted)
				recordEmptyToolTurnAttempt(ctx, bareModel, logfmt.Label(acct.UID, acct.Nickname),
					emptyRetryState.retries+1, "deadline", nil, time.Since(attemptStarted), bufferedStats.TTFB())
				st.status, st.outcome = http.StatusGatewayTimeout, reqlog.OutcomeHTTPError
				releaseHeld()
				return ChatOutcome{Status: http.StatusGatewayTimeout}, &chatFailure{
					Status: http.StatusGatewayTimeout, Code: "upstream_empty_tool_turn",
					Message: "empty tool-turn retry budget expired",
				}
			}
			if errors.Is(readErr, errValidatedResponseTooLarge) {
				credit, hasCredit := bufferedStats.Credit()
				delta := bufferedStats.Usage()
				recordResponsesUsage(h, st, acct.UID, bareModel, delta, credit, hasCredit, attemptStarted)
				st.status, st.outcome = http.StatusBadGateway, reqlog.OutcomeHTTPError
				releaseHeld()
				return ChatOutcome{Status: http.StatusBadGateway}, &chatFailure{
					Status: http.StatusBadGateway, Code: "upstream_validation_limit",
					Message: "upstream response exceeded the 8 MiB validation buffer",
				}
			}
			if readErr != nil {
				credit, hasCredit := bufferedStats.Credit()
				delta := bufferedStats.Usage()
				recordResponsesUsage(h, st, acct.UID, bareModel, delta, credit, hasCredit, attemptStarted)
				st.status, st.outcome = http.StatusBadGateway, reqlog.OutcomeStreamError
				releaseHeld()
				return ChatOutcome{Status: http.StatusBadGateway}, &chatFailure{
					Status: http.StatusBadGateway, Code: "upstream_parse",
					Message: "upstream response could not be buffered for validation",
				}
			}
			rc = io.NopCloser(bytes.NewReader(raw))
			observation, inspectErr := upstream.InspectEmptyToolTurn(raw, req.Body)
			if inspectErr != nil {
				break // renderer 保留原有解析/终态错误语义
			}
			finalObservation = &observation
			if !observation.EmptyToolTurnCandidate() {
				if emptyRetryState.hasRetried() {
					result := "terminal_" + observation.EmptyToolTurnTerminalReason()
					if observation.ValidToolCalls > 0 {
						result = "recovered"
					}
					recordEmptyToolTurnAttempt(ctx, bareModel, logfmt.Label(acct.UID, acct.Nickname),
						emptyRetryState.retries+1, result, observation.Response, time.Since(attemptStarted), bufferedStats.TTFB())
				}
				break
			}

			usage := usageMap(observation.Response)
			if usage != nil {
				st.cacheHit, st.cacheMiss, st.hasCache = 0, 0, false
				if hit, ok := upstream.UsageCacheHitTokens(usage); ok {
					miss, _ := upstream.UsageCacheMissTokens(usage)
					st.cacheHit, st.cacheMiss, st.hasCache = int64(hit), int64(miss), true
				}
				emptyRetryState.noteUsage(observation.Response)
			}
			fillStatFromUsageCompat(st, observation.Response)
			credit, total, hasCredit := usageCreditTotal(observation.Response)
			recordResponsesUsage(h, st, acct.UID, bareModel, usageDeltaFromResponse(observation.Response), credit, hasCredit, attemptStarted)
			if hasCredit {
				h.cfg.Pool.NoteModelCost(acct.UID, bareModel, credit, total)
			}
			hasAlternate := h.hasAlternateEmptyToolTurnAccount(bareModel, realm, tried)
			canRetry, outcome := emptyRetryState.canRetry(time.Now(), hasAlternate)
			attemptNumber := emptyRetryState.retries + 1
			recordEmptyToolTurnAttempt(ctx, bareModel, logfmt.Label(acct.UID, acct.Nickname),
				attemptNumber, outcome, observation.Response, time.Since(attemptStarted), bufferedStats.TTFB())
			if !canRetry {
				applyUsageToChatStat(st, emptyRetryState.usage)
				st.status, st.outcome = http.StatusGatewayTimeout, reqlog.OutcomeHTTPError
				unbindSticky()
				releaseHeld()
				return ChatOutcome{Status: http.StatusGatewayTimeout}, &chatFailure{
					Status: http.StatusGatewayTimeout, Code: "upstream_empty_tool_turn",
					Message: "upstream repeatedly ended a tool-enabled turn without a valid function call",
				}
			}
			emptyRetryState.markRetry()
			releaseHeld()
			unbindSticky()
			retryNextAccount = true
			break
		}
		if retryNextAccount {
			continue
		}
		var uerr *upstream.Error
		if errors.As(terr, &uerr) {
			status = uerr.Status
		}
		if uerr == nil && terr != nil {
			recordResponsesUsage(h, st, acct.UID, bareModel, pool.TokenUsageDelta{}, 0, false, attemptStarted)
			if isUpstreamTimeout(terr, ctx.Err() != nil) {
				st.status = http.StatusServiceUnavailable
				lastErr = fmt.Errorf("%w: %v", errUpstreamTimeout, terr)
				fail(acct.UID)
				break
			}
			st.status = http.StatusServiceUnavailable
			lastErr = terr
			h.cfg.Pool.NoteFailures(acct.UID)
			fail(acct.UID)
			if !rotateBackoff(i, ctx) {
				break
			}
			continue
		}
		if status >= 400 {
			recordResponsesUsage(h, st, acct.UID, bareModel, pool.TokenUsageDelta{}, 0, false, attemptStarted)
			st.status = status
			kind := upstream.ErrNone
			if uerr != nil {
				kind = uerr.Kind
			} else {
				kind = upstream.Classify(status, string(respBody))
				uerr = &upstream.Error{Kind: kind, Status: status, Msg: string(respBody)}
			}
			if req.ResponsesDiagnostics {
				if param, unsupported := responsesUnsupportedUpstreamField(status, respBody, uerr); unsupported {
					st.outcome = reqlog.OutcomeHTTPError
					fail(acct.UID)
					return ChatOutcome{Status: http.StatusBadRequest}, &chatFailure{
						Status:  http.StatusBadRequest,
						Code:    "unsupported_value",
						Message: "The upstream rejected one or more request fields as unsupported.",
						Hint:    responsesUnsupportedFieldHint(req.ResponsesCandidateSections),
						Param:   param,
					}
				}
			}
			if kind == upstream.ErrContentBlocked && (h.cfg.PromptMode == "passthrough" || h.cfg.PromptMode == "append") && !degradedApplied {
				h.degrade.Trigger()
				req.Body = prompt.Rewrite(req.Body, prompt.Degraded)
				degradedApplied = true
				delete(tried, acct.UID)
				releaseHeld()
				log.Printf("WARN: [server] content-blocked (likely fingerprint false positive) -> degraded prompt retry")
				continue
			}
			if kind == upstream.ErrContentBlocked {
				h.applyErrorPolicy(acct.UID, kind, string(respBody), bareModel, uerr)
				fail(acct.UID)
				msg := string(respBody)
				if strings.TrimSpace(msg) == "" {
					msg = "content blocked by upstream content firewall"
				}
				return ChatOutcome{Status: http.StatusBadRequest}, &chatFailure{Status: http.StatusBadRequest, Code: "content_blocked", Message: msg, Hint: h.hintOf(kind, string(respBody), bareModel, reqHasImage, uerr)}
			}
			if kind == upstream.ErrPromptTooLong {
				h.applyErrorPolicy(acct.UID, kind, string(respBody), bareModel, uerr)
				fail(acct.UID)
				return ChatOutcome{Status: http.StatusBadRequest}, &chatFailure{Status: http.StatusBadRequest, Code: "prompt_too_long", Message: promptTooLongMessage(string(respBody)), Hint: h.hintOf(kind, string(respBody), bareModel, reqHasImage, uerr)}
			}
			if kind == upstream.ErrImageInvalid {
				h.applyErrorPolicy(acct.UID, kind, string(respBody), bareModel, uerr)
				fail(acct.UID)
				msg := string(respBody)
				if strings.TrimSpace(msg) == "" {
					msg = "image request was rejected by upstream"
				}
				return ChatOutcome{Status: http.StatusBadRequest}, &chatFailure{Status: http.StatusBadRequest, Code: "image_invalid", Message: msg, Hint: h.hintOf(kind, string(respBody), bareModel, reqHasImage, uerr)}
			}
			lastErr = &upstream.Error{Kind: kind, Status: status, Msg: string(respBody), RetryAfter: uerr.RetryAfter}
			h.applyErrorPolicy(acct.UID, kind, string(respBody), bareModel, uerr)
			fail(acct.UID)
			if kind == upstream.ErrWafBlock && h.wafIP.noteWaf(acct.UID) {
				break
			}
			if !rotateBackoff(i, ctx) {
				break
			}
			continue
		}

		priorUsage := emptyRetryState.usage
		if finalObservation == nil || !finalObservation.SawDone || finalObservation.SawErrorFrame {
			priorUsage = nil
		}
		meta := ChatRenderMeta{Account: acct, BareModel: bareModel, Realm: realm, ReqHasImage: reqHasImage, ChatMeta: chatMeta, Started: st.start, PriorUsage: priorUsage}
		var out ChatOutcome
		if req.Stream {
			out = renderer.RenderStream(ctx, rc, meta)
		} else {
			out = renderer.RenderNonStream(ctx, rc, meta)
		}
		_ = rc.Close()
		if out.Status == 0 {
			out.Status = http.StatusOK
		}
		if bufferedStats != nil {
			out.TTFB = bufferedStats.TTFB()
		}
		// A 200 transport is not sufficient to declare the account healthy: the
		// upstream can emit an error frame or end before [DONE]. Keep the Panel
		// account/cooldown state aligned with the renderer's terminal evidence.
		if out.SawDone && out.UpstreamErr == nil && out.ReadErr == nil {
			h.cfg.Pool.NoteSuccess(acct.UID)
			h.cfg.Pool.BlockModelClear(acct.UID, bareModel)
			if stickyKey != "" && h.cfg.Session != nil {
				h.cfg.Session.Bind(stickyKey, acct.UID)
			}
		}
		st.status = out.Status
		switch {
		case out.Status < 200 || out.Status >= 300:
			st.outcome = reqlog.OutcomeHTTPError
		case out.UpstreamErr != nil || out.ReadErr != nil:
			st.outcome = reqlog.OutcomeStreamError
		case out.RendererErr != nil:
			st.outcome = reqlog.OutcomeInterrupted
		default:
			st.outcome = reqlog.OutcomeSuccess
		}
		st.ttfb = out.TTFB
		st.toks = out.Tokens
		if !out.HasUsage {
			st.toks = -1
		}
		st.promptTokens = int64(out.Prompt)
		st.cacheHit, st.cacheMiss = int64(out.CacheHit), int64(out.CacheMiss)
		st.hasCache = out.CacheHit != 0 || out.CacheMiss != 0 || out.CacheWrite != 0
		st.credit, st.hasCredit = out.Credit, out.HasCredit
		recordedUsage := false
		if resp, ok := out.Response.(map[string]any); ok && resp["object"] != "response" {
			st.toks = completionTokens(resp)
			fillStatFromUsageCompat(st, resp)
			recordResponsesUsage(h, st, acct.UID, bareModel, usageDeltaFromResponse(resp), out.Credit, out.HasCredit, attemptStarted)
			recordedUsage = true
			if credit, total, ok := usageCreditTotal(resp); ok {
				h.cfg.Pool.NoteModelCost(acct.UID, bareModel, credit, total)
			} else if out.HasUsage {
				log.Printf("WARN: [server] response usage without credit acct=%s model=%s (no cost observation)", logfmt.Label(acct.UID, acct.Nickname), bareModel)
			}
		} else if out.HasCredit {
			recordResponsesUsage(h, st, acct.UID, bareModel, responseOutcomeUsage(out), out.Credit, out.HasCredit, attemptStarted)
			recordedUsage = true
			h.cfg.Pool.NoteModelCost(acct.UID, bareModel, out.Credit, out.Prompt+out.Tokens)
		} else if out.HasUsage {
			recordResponsesUsage(h, st, acct.UID, bareModel, responseOutcomeUsage(out), out.Credit, out.HasCredit, attemptStarted)
			recordedUsage = true
			log.Printf("WARN: [server] stream usage without credit acct=%s model=%s (no cost observation)", logfmt.Label(acct.UID, acct.Nickname), bareModel)
		}
		if !recordedUsage {
			recordResponsesUsage(h, st, acct.UID, bareModel, pool.TokenUsageDelta{}, 0, false, attemptStarted)
		}
		if emptyRetryState.hasRetried() && finalObservation != nil && finalObservation.SawDone && !finalObservation.SawErrorFrame {
			applyUsageToChatStat(st, mergeUsageMaps(emptyRetryState.usage, usageMap(finalObservation.Response)))
		}
		return out, nil
	}

	status := http.StatusServiceUnavailable
	code := "no_healthy_account"
	msg := "all accounts are temporarily unavailable, please retry later"
	hint := upstream.NoHealthyAccountHint()
	var ue *upstream.Error
	if errors.As(lastErr, &ue) {
		hint = h.hintOf(ue.Kind, ue.Msg, bareModel, reqHasImage, ue)
		switch ue.Kind {
		case upstream.ErrSoftRate:
			status = http.StatusTooManyRequests
			code = "rate_limit_exceeded"
			msg = "rate limited: all accounts are cooling down, please wait a moment and try again"
		case upstream.ErrWafBlock:
			if h.wafIP.active() {
				code = "waf_ip_blocked"
				msg = "waf ip-level block: upstream firewall is blocking the gateway IP, rotation stopped; retry after the block window expires"
			}
		}
		if s := strings.TrimSpace(ue.Msg); s != "" {
			msg = s
		}
	}
	st.status = status
	return ChatOutcome{Status: status}, &chatFailure{Status: status, Code: code, Message: msg, Hint: hint}
}

type legacyChatRenderer struct {
	h *Handler
	w http.ResponseWriter
}

func (r *legacyChatRenderer) RenderStream(_ context.Context, rc io.ReadCloser, meta ChatRenderMeta) ChatOutcome {
	stats := newChatStatsReaderSince(rc, meta.Started)
	err := upstream.StreamHint(r.w, stats, upstream.FrameHintFunc(func() upstream.HintContext {
		return r.h.hintContext(meta.BareModel, meta.ReqHasImage)
	}))
	out := ChatOutcome{Status: http.StatusOK, TTFB: stats.TTFB()}
	out.Tokens, out.HasUsage = stats.Tokens()
	out.Prompt = stats.promptTokens
	if hit, miss, ok := stats.CacheTokens(); ok {
		out.CacheHit, out.CacheMiss = int(hit), int(miss)
	}
	out.Credit, out.HasCredit = stats.Credit()
	out.SawDone = err == nil
	if upstream.IsEmptyStreamError(err) {
		out.Status = http.StatusBadGateway
		out.ReadErr = err
	}
	if err != nil && !upstream.IsEmptyStreamError(err) {
		out.RendererErr = err
	}
	out.OutputCommitted = true
	return out
}

func (r *legacyChatRenderer) RenderNonStream(_ context.Context, rc io.ReadCloser, _ ChatRenderMeta) ChatOutcome {
	resp, err := upstream.Aggregate(rc)
	if err != nil {
		return ChatOutcome{Status: http.StatusBadGateway, ReadErr: err}
	}
	writeJSON(r.w, http.StatusOK, resp)
	return ChatOutcome{Status: http.StatusOK, Response: resp, OutputCommitted: true}
}

// fillStatFromUsageCompat keeps Responses observations on the same fields used by
// the Panel request log. It accepts the JSON-number shapes produced by decoding
// the Chat aggregate without coupling the protocol adapter to a second schema.
func fillStatFromUsageCompat(st *chatStat, resp map[string]any) {
	if st == nil {
		return
	}
	delta := usageDeltaFromResponse(resp)
	if delta.HasPromptTokens {
		st.promptTokens = delta.PromptTokens
	}
	if delta.HasCompletionTokens {
		st.completionTokens = delta.CompletionTokens
		st.toks = int(delta.CompletionTokens)
	}
	if delta.HasTotalTokens {
		st.totalTokens = delta.TotalTokens
	} else if delta.HasPromptTokens || delta.HasCompletionTokens {
		st.totalTokens = delta.PromptTokens + delta.CompletionTokens
	}
	if u, ok := resp["usage"].(map[string]any); ok {
		if hit, ok := numberInt64(u["prompt_cache_hit_tokens"]); ok {
			st.cacheHit = hit
			st.hasCache = true
		}
		if miss, ok := numberInt64(u["prompt_cache_miss_tokens"]); ok {
			st.cacheMiss = miss
			st.hasCache = true
		}
	}
}

func numberInt64(v any) (int64, bool) {
	switch n := v.(type) {
	case float64:
		return int64(n), true
	case float32:
		return int64(n), true
	case int:
		return int64(n), true
	case int64:
		return n, true
	case json.Number:
		i, err := n.Int64()
		return i, err == nil
	default:
		return 0, false
	}
}

func responseOutcomeUsage(out ChatOutcome) pool.TokenUsageDelta {
	delta := pool.TokenUsageDelta{}
	if out.Prompt >= 0 {
		delta.HasPromptTokens = true
		delta.PromptTokens = int64(out.Prompt)
	}
	if out.Tokens >= 0 {
		delta.HasCompletionTokens = true
		delta.CompletionTokens = int64(out.Tokens)
	}
	if delta.HasPromptTokens || delta.HasCompletionTokens {
		delta.HasTotalTokens = true
		delta.TotalTokens = delta.PromptTokens + delta.CompletionTokens
	}
	return delta
}

func recordResponsesUsage(h *Handler, st *chatStat, uid, model string, delta pool.TokenUsageDelta, credit float64, hasCredit bool, started time.Time) {
	if h == nil || h.cfg.Pool == nil {
		return
	}
	delta.Model = model
	if st != nil {
		st.attempts++
	}
	modelRate := ""
	if h.cfg.Upstream != nil {
		realm, bare := resolveModel(model)
		modelRate = h.cfg.Upstream.ModelRate(realm, bare)
	}
	latency := time.Since(started)
	if latency < time.Millisecond {
		latency = time.Millisecond
	}
	delta.HasLatencyMs = true
	delta.LatencyMs = latency.Milliseconds()
	h.cfg.Pool.RecordTokenUsage(uid, delta)
	if h.cfg.Usage == nil {
		return
	}
	realm := "cn"
	if status, ok := h.cfg.Pool.Status(uid); ok && status.Realm != "" {
		realm = status.Realm
	}
	h.cfg.Usage.Add(time.Now(), realm, uid, model, usage.Delta{
		PromptTokens: delta.PromptTokens, HasPromptTokens: delta.HasPromptTokens,
		CompletionTokens: delta.CompletionTokens, HasCompletion: delta.HasCompletionTokens,
		TotalTokens: delta.TotalTokens, HasTotal: delta.HasTotalTokens,
		Credit: credit, HasCredit: hasCredit,
		CacheHitTokens: st.cacheHit, CacheMissTokens: st.cacheMiss, HasCacheTokens: st.hasCache,
		ModelRate: modelRate,
		LatencyMs: delta.LatencyMs, HasLatency: delta.HasLatencyMs,
	}, delta.HasPromptTokens || delta.HasCompletionTokens || delta.HasTotalTokens)
}

func requestTraceFromContext(ctx context.Context) *requestTrace {
	if ctx == nil {
		return nil
	}
	trace, _ := ctx.Value(requestTraceKey{}).(*requestTrace)
	return trace
}

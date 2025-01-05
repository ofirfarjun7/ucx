package http

/*
#include <stdlib.h>
#include <string.h>
*/
import "C"
import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math/bits"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/openucx/ucx/bindings/go/src/ucx"
)

const AM_REQ = 1
const AM_RESP = 2

func getBuf(buf []byte) (unsafe.Pointer, uint64) {
	if len(buf) > 0 {
		return unsafe.Pointer(&buf[0]), uint64(len(buf))
	}
	return nil, 0
}

type context struct {
	context *ucx.UcpContext
	worker *ucx.UcpWorker
}

func (c *context) Init() {
	contextParams := ucx.UcpParams{}
	contextParams.EnableAM()
	contextParams.EnableStream()
	context, err := ucx.NewUcpContext(&contextParams)
	if err != nil {
		log.Fatalf("Failed to create UCX context: %v", err)
	}

	workerParams := ucx.UcpWorkerParams{}
	workerParams.SetThreadMode(ucx.UCS_THREAD_MODE_MULTI)
	worker, err := context.NewWorker(&workerParams)
	if err != nil {
		log.Fatalf("Failed to create UCX worker: %v", err)
	}

	c.context = context
	c.worker = worker
}

func (c *context) Close() {
	c.worker.Close()
	c.context.Close()
}

func (c *context) Progress() {
	c.worker.Progress();
}

type Server struct {
	context
	listener *ucx.UcpListener
	handler http.Handler
}

type responseWriter struct {
	ep       *ucx.UcpEp
	headers  http.Header
	status   int
	wrote    chan struct{}
	id	 int
	headerSent bool
}

func (w *responseWriter) Header() http.Header {
	return w.headers
}

func (w *responseWriter) onData(request *ucx.UcpRequest, status ucx.UcsStatus) {
	request.Close()
	w.wrote <- struct{}{}
}

func (w *responseWriter) Write(data []byte) (int, error) {
	if !w.headerSent {
		w.WriteHeader(http.StatusOK)
	}
	dataPtr, dataLen := getBuf(data)
	reqParams := &ucx.UcpRequestParams{}
	reqParams.SetCallback(w.onData)
	if _, err := w.ep.SendStreamNonBlocking(w.id, dataPtr, dataLen, reqParams); err != nil {
		return 0, err
	}
	<-w.wrote
	return int(dataLen), nil
}

func (w *responseWriter) WriteHeader(statusCode int) {
	w.status = statusCode

	headerMap := map[string]string{
		"ucx-code": strconv.Itoa(w.status),
		"ucx-id": strconv.Itoa(w.id),
	}

	for k, v := range w.headers {
		headerMap[k] = v[0]
	}

	header, err := json.Marshal(headerMap)
	if err != nil {
		return
	}

	headerPtr, headerLen := getBuf(header)
	_, err = w.ep.SendAmNonBlocking(AM_RESP,
		headerPtr, headerLen, nil, 0,
		ucx.UCP_AM_SEND_FLAG_REPLY, nil)
	if err != nil {
		return
	}
	w.headerSent = true
}

type dataReader struct {
	ep *ucx.UcpEp
	read chan int
	left int
	id int
	onClose func()
}

func (r *dataReader) onData(request *ucx.UcpRequest, status ucx.UcsStatus, length uint64) {
	request.Close()
	r.read <- int(length)
}

func (r *dataReader) Read(p []byte) (length int, res error) {
	if r.left != 0 {
		dataPtr, dataLen := getBuf(p)
		reqParams := &ucx.UcpRequestParams{}
		reqParams.SetCallback(r.onData)
		if _, res = r.ep.RecvStreamNonBlocking(r.id, dataPtr, dataLen, reqParams); res != nil {
			return 0, res
		}
		length = <-r.read
		r.left -= length
		if r.left < 0 {
			log.Fatalf("Read underrun")
		}
	}
	if r.left == 0 {
		res = io.EOF
	}
	return length, res
}

func (r *dataReader) Close() (error) {
	if r.onClose != nil {
		r.onClose()
	}
	return nil
}

func handleAm(header unsafe.Pointer, headerSize uint64, replyEp *ucx.UcpEp) (map[string]string, *dataReader, int64, int, error) {
	var headerMap map[string]string
	if headerSize > 0 {
		headerBytes := ucx.GoBytes(header, headerSize)
		if err := json.Unmarshal(headerBytes, &headerMap); err != nil {
			return nil, nil, 0, 0, err
		}
	}

	length, _ := strconv.ParseInt(headerMap["Content-Length"], 10, 64)
	reqId, _ := strconv.Atoi(headerMap["ucx-id"])
	r := &dataReader {
		ep: replyEp,
		left: int(length),
		id: reqId,
		read: make(chan int, 1),
	}

	return headerMap, r, length, reqId, nil
}

func (s *Server) handleRequest(header unsafe.Pointer, headerSize uint64, data *ucx.UcpAmData, replyEp *ucx.UcpEp) ucx.UcsStatus {
	headerMap, reader, contentLength, reqId, err := handleAm(header, headerSize, replyEp)
	if err != nil {
		fmt.Printf("request %v\n", err)
		return ucx.UCS_ERR_IO_ERROR
	}

	req, _ := http.NewRequest(headerMap["ucx-method"], headerMap["ucx-url"], reader)
	for k, v := range headerMap {
		if !strings.HasPrefix(k, "ucx-") {
			req.Header.Set(k,v)
		}
	}
	req.ContentLength = contentLength
	req.RequestURI = req.URL.EscapedPath()
	writer := &responseWriter{
		ep: replyEp,
		headers: make(http.Header),
		id: reqId,
		wrote: make(chan struct{}, 1),
	}

	go s.handler.ServeHTTP(writer, req)
	return ucx.UCS_OK
}

func onErr(ep *ucx.UcpEp, status ucx.UcsStatus) {
	if status == ucx.UCS_ERR_CONNECTION_RESET {
		return
	}
	errorString := fmt.Sprintf("Endpoint error: %v", status.String())
	panic(errorString)
}

func (s *Server) servConn(conn *ucx.UcpConnectionRequest) {
	epParams := &ucx.UcpEpParams{}
	epParams.SetConnRequest(conn)
	epParams.SetErrorHandler(onErr)
	_, err := s.worker.NewEndpoint(epParams)
	if err != nil {
		fmt.Printf("Failed to create endpoint: %v\n", err)
		return
	}
}

func (s *Server) Close() {
	s.listener.Close()
	s.context.Close()
}

func NewServer(addr string, handler http.Handler) (*Server, error) {
	s := &Server{
		handler: handler,
	}
	s.Init()

	s.worker.SetAmRecvHandler(AM_REQ, ucx.UCP_AM_FLAG_PERSISTENT_DATA, s.handleRequest)

	tcp, err := net.ResolveTCPAddr("tcp", addr)
	if err != nil {
		return nil, err
	}

	listenerParams := &ucx.UcpListenerParams{}
	listenerParams.SetSocketAddress(tcp)
	listenerParams.SetConnectionHandler(s.servConn)
	listener, err := s.worker.NewListener(listenerParams);
	if err != nil {
		return nil, err
	}

	s.listener = listener
	return s, nil
}

func (s *Server) Serve() {
	for {
		s.Progress();
	}
}

func StartServer(addr string, handler http.Handler) (serve func() error, err error) {
	s, err := NewServer(addr, handler)
	if (err != nil) {
		return nil, err
	}
	serve = func() error {
		s.Serve()
		return nil
	}
	return
}

const (
	TR_PENDING_SEND = 1 << iota
	TR_PENDING_RECV
)

type tracker struct {
	resp chan *http.Response
	noBody bool
	pending int
	data []byte
}

type Transport struct {
	context
	ep *ucx.UcpEp
	chmap uint64
	reqs sync.Map
	quit chan struct{}
}

func (a *Transport) handleResponse(header unsafe.Pointer, headerSize uint64, data *ucx.UcpAmData, replyEp *ucx.UcpEp) ucx.UcsStatus {
	headerMap, reader, contentLength, reqId, err := handleAm(header, headerSize, replyEp)
	if err != nil {
		fmt.Printf("handleResponse %v\n", err)
		return ucx.UCS_ERR_IO_ERROR
	}

	req, _ := a.reqs.Load(reqId)
	tr := req.(*tracker)
	if tr.noBody {
		reader.left = 0
	}

	resp := &http.Response{
		Header: make(http.Header),
		Body: reader,
		Status: headerMap["ucx-code"],
	}

	reader.onClose = func() {
		a.donePending(tr, TR_PENDING_RECV, reqId)
	}

	statusCode, _, _ := strings.Cut(resp.Status, " ")
	resp.StatusCode, _ = strconv.Atoi(statusCode)
	resp.ContentLength = contentLength

	for k, v := range headerMap {
		resp.Header.Set(k,v)
	}

	tr.resp <- resp
	return ucx.UCS_OK
}

func (a *Transport) Close() {
	a.quit <- struct{}{}
	a.ep.CloseNonBlockingForce(nil)
	a.context.Close()
}

func (a *Transport) progress() {
	for {
		select {
			case <-a.quit: return
			default: a.Progress()
		}
	}
}

func NewTransport(addr string) (*Transport, error) {
	a := new(Transport)
	a.Init()
	a.quit = make(chan struct{})
	a.chmap = ^uint64(0)

	a.worker.SetAmRecvHandler(AM_RESP, ucx.UCP_AM_FLAG_PERSISTENT_DATA, a.handleResponse)

	tcp, err := net.ResolveTCPAddr("tcp", addr)
	if err != nil {
		return nil, err
	}

	epParams := &ucx.UcpEpParams{}
	epParams.SetSocketAddress(tcp)
	epParams.SetErrorHandler(onErr)
	ep, err := a.worker.NewEndpoint(epParams)
	if err != nil {
		return nil, err
	}

	a.ep = ep
	go a.progress()
	return a, nil
}

func (a *Transport) getCh() (int, bool) {
    for {
        m := atomic.LoadUint64(&a.chmap)
        if m == 0 {
            return -1, false
        }

        chId := bits.TrailingZeros64(m)
        n := m &^ (uint64(1) << chId)

        if atomic.CompareAndSwapUint64(&a.chmap, m, n) {
            return chId, true
        }
    }
}

func (a *Transport) putCh(chId int) {
    for {
        m := atomic.LoadUint64(&a.chmap)
        n := m | (uint64(1) << chId)

        if atomic.CompareAndSwapUint64(&a.chmap, m, n) {
            return
        }
    }
}

func (a *Transport) donePending(tr *tracker, f int, reqId int) {
	tr.pending &= ^f
	if tr.pending == 0 {
		a.putCh(reqId)
	}
}

func (a *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	reqId, ok := a.getCh()
	if !ok {
		return nil, errors.New("Can't allocate channel")
	}
	headerMap := map[string]string{
		"ucx-method": req.Method,
		"ucx-url": req.URL.String(),
		"ucx-id": strconv.Itoa(reqId),
		"Content-Length": strconv.FormatInt(req.ContentLength, 10),
	}

	for k, v := range req.Header {
		headerMap[k] = v[0]
	}

	header, err := json.Marshal(headerMap)
	if err != nil {
		return nil, err
	}

	tr := &tracker{
		resp: make(chan *http.Response),
		noBody: req.Method == http.MethodHead,
		pending: TR_PENDING_RECV,
	}
	a.reqs.Store(reqId, tr)

	headerPtr, headerLen := getBuf(header)
	send, err := a.ep.SendAmNonBlocking(AM_REQ,
		headerPtr, headerLen, nil, 0,
		ucx.UCP_AM_SEND_FLAG_REPLY, nil)
	if err != nil {
		return nil, err
	}
	defer send.Close()

	if req.Body != nil {
		tr.data, err = ioutil.ReadAll(req.Body)
		dataPtr, dataLen := getBuf(tr.data)
		reqParams := &ucx.UcpRequestParams{}
		reqParams.SetCallback(func (request *ucx.UcpRequest, status ucx.UcsStatus) {
			a.donePending(tr, TR_PENDING_SEND, reqId)
		})
		tr.pending |= TR_PENDING_SEND
		req, err := a.ep.SendStreamNonBlocking(reqId, dataPtr, dataLen, reqParams)
		if err != nil {
			return nil, err
		}
		req.Close()
	}

	resp := <-tr.resp
	a.reqs.Delete(reqId)
	return resp, nil
}

func Dump(o interface{}) string {
	switch o := o.(type) {
	case *dataReader:
		return fmt.Sprintf("dataReader %d %d", o.left, o.id)
	}
	return fmt.Sprintf("%T %v", o, o)
}

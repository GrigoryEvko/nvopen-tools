// Function: ctor_410
// Address: 0x52e800
//
int ctor_410()
{
  int v0; // edx
  __int64 v1; // rax
  __int64 v2; // rdx
  int v3; // edx
  __int64 v4; // rax
  __int64 v5; // rdx
  int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // edx
  __int64 v13; // rbx
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v17; // [rsp+0h] [rbp-60h]
  __int64 v18; // [rsp+0h] [rbp-60h]
  __int64 v19; // [rsp+0h] [rbp-60h]
  __int64 v20; // [rsp+0h] [rbp-60h]
  char v21; // [rsp+13h] [rbp-4Dh] BYREF
  int v22; // [rsp+14h] [rbp-4Ch] BYREF
  char *v23; // [rsp+18h] [rbp-48h] BYREF
  const char *v24; // [rsp+20h] [rbp-40h] BYREF
  __int64 v25; // [rsp+28h] [rbp-38h]

  v24 = "Instrument memory accesses";
  v23 = &v21;
  v22 = 1;
  v25 = 26;
  v21 = 1;
  sub_24CC910(&unk_4FEE300, "tsan-instrument-memory-accesses", &v23, &v24, &v22);
  __cxa_atexit(sub_984900, &unk_4FEE300, &qword_4A427C0);
  v23 = &v21;
  v24 = "Instrument function entry and exit";
  v22 = 1;
  v25 = 34;
  v21 = 1;
  sub_24CC910(&unk_4FEE220, "tsan-instrument-func-entry-exit", &v23, &v24, &v22);
  __cxa_atexit(sub_984900, &unk_4FEE220, &qword_4A427C0);
  qword_4FEE140 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FEE14C &= 0x8000u;
  word_4FEE150 = 0;
  qword_4FEE190 = 0x100000000LL;
  qword_4FEE158 = 0;
  qword_4FEE160 = 0;
  qword_4FEE168 = 0;
  dword_4FEE148 = v0;
  qword_4FEE170 = 0;
  qword_4FEE178 = 0;
  qword_4FEE180 = 0;
  qword_4FEE188 = (__int64)&unk_4FEE198;
  qword_4FEE1A0 = 0;
  qword_4FEE1A8 = (__int64)&unk_4FEE1C0;
  qword_4FEE1B0 = 1;
  dword_4FEE1B8 = 0;
  byte_4FEE1BC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FEE190;
  if ( (unsigned __int64)(unsigned int)qword_4FEE190 + 1 > HIDWORD(qword_4FEE190) )
  {
    v17 = v1;
    sub_C8D5F0((char *)&unk_4FEE198 - 16, &unk_4FEE198, (unsigned int)qword_4FEE190 + 1LL, 8);
    v2 = (unsigned int)qword_4FEE190;
    v1 = v17;
  }
  *(_QWORD *)(qword_4FEE188 + 8 * v2) = v1;
  LODWORD(qword_4FEE190) = qword_4FEE190 + 1;
  qword_4FEE1C8 = 0;
  qword_4FEE1D0 = (__int64)&unk_49D9748;
  qword_4FEE1D8 = 0;
  qword_4FEE140 = (__int64)&unk_49DC090;
  qword_4FEE1E0 = (__int64)&unk_49DC1D0;
  qword_4FEE200 = (__int64)nullsub_23;
  qword_4FEE1F8 = (__int64)sub_984030;
  sub_C53080(&qword_4FEE140, "tsan-handle-cxx-exceptions", 26);
  qword_4FEE168 = (__int64)"Handle C++ exceptions (insert cleanup blocks for unwinding)";
  LOWORD(qword_4FEE1D8) = 257;
  LOBYTE(qword_4FEE1C8) = 1;
  qword_4FEE170 = 59;
  LOBYTE(dword_4FEE14C) = dword_4FEE14C & 0x9F | 0x20;
  sub_C53130(&qword_4FEE140);
  __cxa_atexit(sub_984900, &qword_4FEE140, &qword_4A427C0);
  qword_4FEE060 = (__int64)&unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FEE0B0 = 0x100000000LL;
  dword_4FEE06C &= 0x8000u;
  qword_4FEE0A8 = (__int64)&unk_4FEE0B8;
  word_4FEE070 = 0;
  qword_4FEE078 = 0;
  dword_4FEE068 = v3;
  qword_4FEE080 = 0;
  qword_4FEE088 = 0;
  qword_4FEE090 = 0;
  qword_4FEE098 = 0;
  qword_4FEE0A0 = 0;
  qword_4FEE0C0 = 0;
  qword_4FEE0C8 = (__int64)&unk_4FEE0E0;
  qword_4FEE0D0 = 1;
  dword_4FEE0D8 = 0;
  byte_4FEE0DC = 1;
  v4 = sub_C57470();
  v5 = (unsigned int)qword_4FEE0B0;
  if ( (unsigned __int64)(unsigned int)qword_4FEE0B0 + 1 > HIDWORD(qword_4FEE0B0) )
  {
    v18 = v4;
    sub_C8D5F0((char *)&unk_4FEE0B8 - 16, &unk_4FEE0B8, (unsigned int)qword_4FEE0B0 + 1LL, 8);
    v5 = (unsigned int)qword_4FEE0B0;
    v4 = v18;
  }
  *(_QWORD *)(qword_4FEE0A8 + 8 * v5) = v4;
  LODWORD(qword_4FEE0B0) = qword_4FEE0B0 + 1;
  qword_4FEE0E8 = 0;
  qword_4FEE0F0 = (__int64)&unk_49D9748;
  qword_4FEE0F8 = 0;
  qword_4FEE060 = (__int64)&unk_49DC090;
  qword_4FEE100 = (__int64)&unk_49DC1D0;
  qword_4FEE120 = (__int64)nullsub_23;
  qword_4FEE118 = (__int64)sub_984030;
  sub_C53080(&qword_4FEE060, "tsan-instrument-atomics", 23);
  qword_4FEE088 = (__int64)"Instrument atomics";
  LOWORD(qword_4FEE0F8) = 257;
  LOBYTE(qword_4FEE0E8) = 1;
  qword_4FEE090 = 18;
  LOBYTE(dword_4FEE06C) = dword_4FEE06C & 0x9F | 0x20;
  sub_C53130(&qword_4FEE060);
  __cxa_atexit(sub_984900, &qword_4FEE060, &qword_4A427C0);
  qword_4FEDF80 = (__int64)&unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FEDFD0 = 0x100000000LL;
  dword_4FEDF8C &= 0x8000u;
  qword_4FEDFC8 = (__int64)&unk_4FEDFD8;
  word_4FEDF90 = 0;
  qword_4FEDF98 = 0;
  dword_4FEDF88 = v6;
  qword_4FEDFA0 = 0;
  qword_4FEDFA8 = 0;
  qword_4FEDFB0 = 0;
  qword_4FEDFB8 = 0;
  qword_4FEDFC0 = 0;
  qword_4FEDFE0 = 0;
  qword_4FEDFE8 = (__int64)&unk_4FEE000;
  qword_4FEDFF0 = 1;
  dword_4FEDFF8 = 0;
  byte_4FEDFFC = 1;
  v7 = sub_C57470();
  v8 = (unsigned int)qword_4FEDFD0;
  if ( (unsigned __int64)(unsigned int)qword_4FEDFD0 + 1 > HIDWORD(qword_4FEDFD0) )
  {
    v19 = v7;
    sub_C8D5F0((char *)&unk_4FEDFD8 - 16, &unk_4FEDFD8, (unsigned int)qword_4FEDFD0 + 1LL, 8);
    v8 = (unsigned int)qword_4FEDFD0;
    v7 = v19;
  }
  *(_QWORD *)(qword_4FEDFC8 + 8 * v8) = v7;
  LODWORD(qword_4FEDFD0) = qword_4FEDFD0 + 1;
  qword_4FEE008 = 0;
  qword_4FEE010 = (__int64)&unk_49D9748;
  qword_4FEE018 = 0;
  qword_4FEDF80 = (__int64)&unk_49DC090;
  qword_4FEE020 = (__int64)&unk_49DC1D0;
  qword_4FEE040 = (__int64)nullsub_23;
  qword_4FEE038 = (__int64)sub_984030;
  sub_C53080(&qword_4FEDF80, "tsan-instrument-memintrinsics", 29);
  qword_4FEDFA8 = (__int64)"Instrument memintrinsics (memset/memcpy/memmove)";
  LOWORD(qword_4FEE018) = 257;
  LOBYTE(qword_4FEE008) = 1;
  qword_4FEDFB0 = 48;
  LOBYTE(dword_4FEDF8C) = dword_4FEDF8C & 0x9F | 0x20;
  sub_C53130(&qword_4FEDF80);
  __cxa_atexit(sub_984900, &qword_4FEDF80, &qword_4A427C0);
  qword_4FEDEA0 = (__int64)&unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FEDEF0 = 0x100000000LL;
  dword_4FEDEAC &= 0x8000u;
  qword_4FEDEE8 = (__int64)&unk_4FEDEF8;
  word_4FEDEB0 = 0;
  qword_4FEDEB8 = 0;
  dword_4FEDEA8 = v9;
  qword_4FEDEC0 = 0;
  qword_4FEDEC8 = 0;
  qword_4FEDED0 = 0;
  qword_4FEDED8 = 0;
  qword_4FEDEE0 = 0;
  qword_4FEDF00 = 0;
  qword_4FEDF08 = (__int64)&unk_4FEDF20;
  qword_4FEDF10 = 1;
  dword_4FEDF18 = 0;
  byte_4FEDF1C = 1;
  v10 = sub_C57470();
  v11 = (unsigned int)qword_4FEDEF0;
  if ( (unsigned __int64)(unsigned int)qword_4FEDEF0 + 1 > HIDWORD(qword_4FEDEF0) )
  {
    v20 = v10;
    sub_C8D5F0((char *)&unk_4FEDEF8 - 16, &unk_4FEDEF8, (unsigned int)qword_4FEDEF0 + 1LL, 8);
    v11 = (unsigned int)qword_4FEDEF0;
    v10 = v20;
  }
  *(_QWORD *)(qword_4FEDEE8 + 8 * v11) = v10;
  LODWORD(qword_4FEDEF0) = qword_4FEDEF0 + 1;
  qword_4FEDF28 = 0;
  qword_4FEDF30 = (__int64)&unk_49D9748;
  qword_4FEDF38 = 0;
  qword_4FEDEA0 = (__int64)&unk_49DC090;
  qword_4FEDF40 = (__int64)&unk_49DC1D0;
  qword_4FEDF60 = (__int64)nullsub_23;
  qword_4FEDF58 = (__int64)sub_984030;
  sub_C53080(&qword_4FEDEA0, "tsan-distinguish-volatile", 25);
  qword_4FEDEC8 = (__int64)"Emit special instrumentation for accesses to volatiles";
  LOWORD(qword_4FEDF38) = 256;
  LOBYTE(qword_4FEDF28) = 0;
  qword_4FEDED0 = 54;
  LOBYTE(dword_4FEDEAC) = dword_4FEDEAC & 0x9F | 0x20;
  sub_C53130(&qword_4FEDEA0);
  __cxa_atexit(sub_984900, &qword_4FEDEA0, &qword_4A427C0);
  qword_4FEDDC0 = (__int64)&unk_49DC150;
  v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FEDE10 = 0x100000000LL;
  dword_4FEDDCC &= 0x8000u;
  word_4FEDDD0 = 0;
  qword_4FEDE08 = (__int64)&unk_4FEDE18;
  qword_4FEDDD8 = 0;
  dword_4FEDDC8 = v12;
  qword_4FEDDE0 = 0;
  qword_4FEDDE8 = 0;
  qword_4FEDDF0 = 0;
  qword_4FEDDF8 = 0;
  qword_4FEDE00 = 0;
  qword_4FEDE20 = 0;
  qword_4FEDE28 = (__int64)&unk_4FEDE40;
  qword_4FEDE30 = 1;
  dword_4FEDE38 = 0;
  byte_4FEDE3C = 1;
  v13 = sub_C57470();
  v14 = (unsigned int)qword_4FEDE10;
  v15 = (unsigned int)qword_4FEDE10 + 1LL;
  if ( v15 > HIDWORD(qword_4FEDE10) )
  {
    sub_C8D5F0((char *)&unk_4FEDE18 - 16, &unk_4FEDE18, v15, 8);
    v14 = (unsigned int)qword_4FEDE10;
  }
  *(_QWORD *)(qword_4FEDE08 + 8 * v14) = v13;
  LODWORD(qword_4FEDE10) = qword_4FEDE10 + 1;
  qword_4FEDE48 = 0;
  qword_4FEDE50 = (__int64)&unk_49D9748;
  qword_4FEDE58 = 0;
  qword_4FEDDC0 = (__int64)&unk_49DC090;
  qword_4FEDE60 = (__int64)&unk_49DC1D0;
  qword_4FEDE80 = (__int64)nullsub_23;
  qword_4FEDE78 = (__int64)sub_984030;
  sub_C53080(&qword_4FEDDC0, "tsan-instrument-read-before-write", 33);
  LOBYTE(qword_4FEDE48) = 0;
  LOWORD(qword_4FEDE58) = 256;
  qword_4FEDDE8 = (__int64)"Do not eliminate read instrumentation for read-before-writes";
  qword_4FEDDF0 = 60;
  LOBYTE(dword_4FEDDCC) = dword_4FEDDCC & 0x9F | 0x20;
  sub_C53130(&qword_4FEDDC0);
  __cxa_atexit(sub_984900, &qword_4FEDDC0, &qword_4A427C0);
  v22 = 1;
  v24 = "Emit special compound instrumentation for reads-before-writes";
  v25 = 61;
  v21 = 0;
  v23 = &v21;
  sub_24CC910(&unk_4FEDCE0, "tsan-compound-read-before-write", &v23, &v24, &v22);
  return __cxa_atexit(sub_984900, &unk_4FEDCE0, &qword_4A427C0);
}

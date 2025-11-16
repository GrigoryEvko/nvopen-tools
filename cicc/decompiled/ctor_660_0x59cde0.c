// Function: ctor_660
// Address: 0x59cde0
//
int __fastcall ctor_660(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // edx
  __int64 v17; // r12
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rcx
  int v22; // edx
  __int64 v23; // rbx
  __int64 v24; // rax
  unsigned __int64 v25; // rdx

  qword_503A3A0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_503A3F0 = 0x100000000LL;
  dword_503A3AC &= 0x8000u;
  word_503A3B0 = 0;
  qword_503A3B8 = 0;
  qword_503A3C0 = 0;
  dword_503A3A8 = v4;
  qword_503A3C8 = 0;
  qword_503A3D0 = 0;
  qword_503A3D8 = 0;
  qword_503A3E0 = 0;
  qword_503A3E8 = (__int64)&unk_503A3F8;
  qword_503A400 = 0;
  qword_503A408 = (__int64)&unk_503A420;
  qword_503A410 = 1;
  dword_503A418 = 0;
  byte_503A41C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_503A3F0;
  v7 = (unsigned int)qword_503A3F0 + 1LL;
  if ( v7 > HIDWORD(qword_503A3F0) )
  {
    sub_C8D5F0((char *)&unk_503A3F8 - 16, &unk_503A3F8, v7, 8);
    v6 = (unsigned int)qword_503A3F0;
  }
  *(_QWORD *)(qword_503A3E8 + 8 * v6) = v5;
  LODWORD(qword_503A3F0) = qword_503A3F0 + 1;
  qword_503A428 = 0;
  qword_503A430 = (__int64)&unk_49D9748;
  qword_503A438 = 0;
  qword_503A3A0 = (__int64)&unk_49DC090;
  qword_503A440 = (__int64)&unk_49DC1D0;
  qword_503A460 = (__int64)nullsub_23;
  qword_503A458 = (__int64)sub_984030;
  sub_C53080(&qword_503A3A0, "force-instr-ref-livedebugvalues", 31);
  LOWORD(qword_503A438) = 256;
  LOBYTE(qword_503A428) = 0;
  qword_503A3D0 = 70;
  LOBYTE(dword_503A3AC) = dword_503A3AC & 0x9F | 0x20;
  qword_503A3C8 = (__int64)"Use instruction-ref based LiveDebugValues with normal DBG_VALUE inputs";
  sub_C53130(&qword_503A3A0);
  __cxa_atexit(sub_984900, &qword_503A3A0, &qword_4A427C0);
  qword_503A2C0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_503A3A0, v8, v9), 1u);
  qword_503A310 = 0x100000000LL;
  dword_503A2CC &= 0x8000u;
  word_503A2D0 = 0;
  qword_503A2D8 = 0;
  qword_503A2E0 = 0;
  dword_503A2C8 = v10;
  qword_503A2E8 = 0;
  qword_503A2F0 = 0;
  qword_503A2F8 = 0;
  qword_503A300 = 0;
  qword_503A308 = (__int64)&unk_503A318;
  qword_503A320 = 0;
  qword_503A328 = (__int64)&unk_503A340;
  qword_503A330 = 1;
  dword_503A338 = 0;
  byte_503A33C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_503A310;
  v13 = (unsigned int)qword_503A310 + 1LL;
  if ( v13 > HIDWORD(qword_503A310) )
  {
    sub_C8D5F0((char *)&unk_503A318 - 16, &unk_503A318, v13, 8);
    v12 = (unsigned int)qword_503A310;
  }
  *(_QWORD *)(qword_503A308 + 8 * v12) = v11;
  LODWORD(qword_503A310) = qword_503A310 + 1;
  qword_503A348 = 0;
  qword_503A350 = (__int64)&unk_49DC110;
  qword_503A358 = 0;
  qword_503A2C0 = (__int64)&unk_49D97F0;
  qword_503A360 = (__int64)&unk_49DC200;
  qword_503A380 = (__int64)nullsub_26;
  qword_503A378 = (__int64)sub_9C26D0;
  sub_C53080(&qword_503A2C0, "experimental-debug-variable-locations", 37);
  qword_503A2F0 = 54;
  qword_503A2E8 = (__int64)"Use experimental new value-tracking variable locations";
  sub_C53130(&qword_503A2C0);
  __cxa_atexit(sub_9C44F0, &qword_503A2C0, &qword_4A427C0);
  qword_503A1E0 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_9C44F0, &qword_503A2C0, v14, v15), 1u);
  byte_503A25C = 1;
  qword_503A230 = 0x100000000LL;
  dword_503A1EC &= 0x8000u;
  qword_503A1F8 = 0;
  qword_503A200 = 0;
  qword_503A208 = 0;
  dword_503A1E8 = v16;
  word_503A1F0 = 0;
  qword_503A210 = 0;
  qword_503A218 = 0;
  qword_503A220 = 0;
  qword_503A228 = (__int64)&unk_503A238;
  qword_503A240 = 0;
  qword_503A248 = (__int64)&unk_503A260;
  qword_503A250 = 1;
  dword_503A258 = 0;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_503A230;
  v19 = (unsigned int)qword_503A230 + 1LL;
  if ( v19 > HIDWORD(qword_503A230) )
  {
    sub_C8D5F0((char *)&unk_503A238 - 16, &unk_503A238, v19, 8);
    v18 = (unsigned int)qword_503A230;
  }
  *(_QWORD *)(qword_503A228 + 8 * v18) = v17;
  qword_503A270 = (__int64)&unk_49D9728;
  qword_503A1E0 = (__int64)&unk_49DBF10;
  LODWORD(qword_503A230) = qword_503A230 + 1;
  qword_503A268 = 0;
  qword_503A280 = (__int64)&unk_49DC290;
  qword_503A278 = 0;
  qword_503A2A0 = (__int64)nullsub_24;
  qword_503A298 = (__int64)sub_984050;
  sub_C53080(&qword_503A1E0, "livedebugvalues-input-bb-limit", 30);
  qword_503A210 = 57;
  qword_503A208 = (__int64)"Maximum input basic blocks before DBG_VALUE limit applies";
  LODWORD(qword_503A268) = 10000;
  BYTE4(qword_503A278) = 1;
  LODWORD(qword_503A278) = 10000;
  LOBYTE(dword_503A1EC) = dword_503A1EC & 0x9F | 0x20;
  sub_C53130(&qword_503A1E0);
  __cxa_atexit(sub_984970, &qword_503A1E0, &qword_4A427C0);
  qword_503A100 = (__int64)&unk_49DC150;
  v22 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_503A1E0, v20, v21), 1u);
  dword_503A10C &= 0x8000u;
  word_503A110 = 0;
  qword_503A150 = 0x100000000LL;
  qword_503A118 = 0;
  qword_503A120 = 0;
  qword_503A128 = 0;
  dword_503A108 = v22;
  qword_503A130 = 0;
  qword_503A138 = 0;
  qword_503A140 = 0;
  qword_503A148 = (__int64)&unk_503A158;
  qword_503A160 = 0;
  qword_503A168 = (__int64)&unk_503A180;
  qword_503A170 = 1;
  dword_503A178 = 0;
  byte_503A17C = 1;
  v23 = sub_C57470();
  v24 = (unsigned int)qword_503A150;
  v25 = (unsigned int)qword_503A150 + 1LL;
  if ( v25 > HIDWORD(qword_503A150) )
  {
    sub_C8D5F0((char *)&unk_503A158 - 16, &unk_503A158, v25, 8);
    v24 = (unsigned int)qword_503A150;
  }
  *(_QWORD *)(qword_503A148 + 8 * v24) = v23;
  qword_503A190 = (__int64)&unk_49D9728;
  qword_503A100 = (__int64)&unk_49DBF10;
  LODWORD(qword_503A150) = qword_503A150 + 1;
  qword_503A188 = 0;
  qword_503A1A0 = (__int64)&unk_49DC290;
  qword_503A198 = 0;
  qword_503A1C0 = (__int64)nullsub_24;
  qword_503A1B8 = (__int64)sub_984050;
  sub_C53080(&qword_503A100, "livedebugvalues-input-dbg-value-limit", 37);
  qword_503A130 = 64;
  qword_503A128 = (__int64)"Maximum input DBG_VALUE insts supported by debug range extension";
  LODWORD(qword_503A188) = 50000;
  BYTE4(qword_503A198) = 1;
  LODWORD(qword_503A198) = 50000;
  LOBYTE(dword_503A10C) = dword_503A10C & 0x9F | 0x20;
  sub_C53130(&qword_503A100);
  return __cxa_atexit(sub_984970, &qword_503A100, &qword_4A427C0);
}

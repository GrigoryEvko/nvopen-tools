// Function: ctor_709
// Address: 0x5be630
//
_QWORD *__fastcall ctor_709(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx

  qword_5051260 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_50512B0 = 0x100000000LL;
  word_5051270 = 0;
  dword_505126C &= 0x8000u;
  qword_5051278 = 0;
  qword_5051280 = 0;
  dword_5051268 = v4;
  qword_5051288 = 0;
  qword_5051290 = 0;
  qword_5051298 = 0;
  qword_50512A0 = 0;
  qword_50512A8 = (__int64)&unk_50512B8;
  qword_50512C0 = 0;
  qword_50512C8 = (__int64)&unk_50512E0;
  qword_50512D0 = 1;
  dword_50512D8 = 0;
  byte_50512DC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50512B0;
  v7 = (unsigned int)qword_50512B0 + 1LL;
  if ( v7 > HIDWORD(qword_50512B0) )
  {
    sub_C8D5F0((char *)&unk_50512B8 - 16, &unk_50512B8, v7, 8);
    v6 = (unsigned int)qword_50512B0;
  }
  *(_QWORD *)(qword_50512A8 + 8 * v6) = v5;
  LODWORD(qword_50512B0) = qword_50512B0 + 1;
  qword_50512E8 = 0;
  qword_50512F0 = (__int64)&unk_49D9748;
  qword_50512F8 = 0;
  qword_5051260 = (__int64)&unk_49DC090;
  qword_5051300 = (__int64)&unk_49DC1D0;
  qword_5051320 = (__int64)nullsub_23;
  qword_5051318 = (__int64)sub_984030;
  sub_C53080(&qword_5051260, "emulate-old-livedebugvalues", 27);
  qword_5051290 = 32;
  LOBYTE(qword_50512E8) = 0;
  LOBYTE(dword_505126C) = dword_505126C & 0x9F | 0x20;
  qword_5051288 = (__int64)"Act like old LiveDebugValues did";
  LOWORD(qword_50512F8) = 256;
  sub_C53130(&qword_5051260);
  __cxa_atexit(sub_984900, &qword_5051260, &qword_4A427C0);
  qword_5051180 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5051260, v8, v9), 1u);
  byte_50511FC = 1;
  qword_50511D0 = 0x100000000LL;
  dword_505118C &= 0x8000u;
  qword_5051198 = 0;
  qword_50511A0 = 0;
  qword_50511A8 = 0;
  dword_5051188 = v10;
  word_5051190 = 0;
  qword_50511B0 = 0;
  qword_50511B8 = 0;
  qword_50511C0 = 0;
  qword_50511C8 = (__int64)&unk_50511D8;
  qword_50511E0 = 0;
  qword_50511E8 = (__int64)&unk_5051200;
  qword_50511F0 = 1;
  dword_50511F8 = 0;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_50511D0;
  v13 = (unsigned int)qword_50511D0 + 1LL;
  if ( v13 > HIDWORD(qword_50511D0) )
  {
    sub_C8D5F0((char *)&unk_50511D8 - 16, &unk_50511D8, v13, 8);
    v12 = (unsigned int)qword_50511D0;
  }
  *(_QWORD *)(qword_50511C8 + 8 * v12) = v11;
  LODWORD(qword_50511D0) = qword_50511D0 + 1;
  qword_5051208 = 0;
  qword_5051210 = (__int64)&unk_49D9728;
  qword_5051218 = 0;
  qword_5051180 = (__int64)&unk_49DBF10;
  qword_5051220 = (__int64)&unk_49DC290;
  qword_5051240 = (__int64)nullsub_24;
  qword_5051238 = (__int64)sub_984050;
  sub_C53080(&qword_5051180, "livedebugvalues-max-stack-slots", 31);
  qword_50511B0 = 30;
  LODWORD(qword_5051208) = 250;
  BYTE4(qword_5051218) = 1;
  LODWORD(qword_5051218) = 250;
  LOBYTE(dword_505118C) = dword_505118C & 0x9F | 0x20;
  qword_50511A8 = (__int64)"livedebugvalues-stack-ws-limit";
  sub_C53130(&qword_5051180);
  __cxa_atexit(sub_984970, &qword_5051180, &qword_4A427C0);
  unk_5051178 = -1;
  unk_5051170 = -1;
  qword_5051168 = 0xFFFFFEFFFFFFFFFFLL;
  return &qword_5051168;
}

// Function: ctor_081
// Address: 0x49e180
//
int ctor_081()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // r12
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // rbx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v19; // [rsp+8h] [rbp-38h]
  __int64 v20; // [rsp+8h] [rbp-38h]

  qword_4F8F200 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8F250 = 0x100000000LL;
  dword_4F8F20C &= 0x8000u;
  word_4F8F210 = 0;
  qword_4F8F218 = 0;
  qword_4F8F220 = 0;
  dword_4F8F208 = v0;
  qword_4F8F228 = 0;
  qword_4F8F230 = 0;
  qword_4F8F238 = 0;
  qword_4F8F240 = 0;
  qword_4F8F248 = (__int64)&unk_4F8F258;
  qword_4F8F260 = 0;
  qword_4F8F268 = (__int64)&unk_4F8F280;
  qword_4F8F270 = 1;
  dword_4F8F278 = 0;
  byte_4F8F27C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F8F250;
  v3 = (unsigned int)qword_4F8F250 + 1LL;
  if ( v3 > HIDWORD(qword_4F8F250) )
  {
    sub_C8D5F0((char *)&unk_4F8F258 - 16, &unk_4F8F258, v3, 8);
    v2 = (unsigned int)qword_4F8F250;
  }
  *(_QWORD *)(qword_4F8F248 + 8 * v2) = v1;
  qword_4F8F290 = (__int64)&unk_49D9728;
  qword_4F8F200 = (__int64)&unk_49DBF10;
  LODWORD(qword_4F8F250) = qword_4F8F250 + 1;
  qword_4F8F288 = 0;
  qword_4F8F2A0 = (__int64)&unk_49DC290;
  qword_4F8F298 = 0;
  qword_4F8F2C0 = (__int64)nullsub_24;
  qword_4F8F2B8 = (__int64)sub_984050;
  sub_C53080(&qword_4F8F200, "memdep-block-scan-limit", 23);
  LODWORD(qword_4F8F288) = 100;
  BYTE4(qword_4F8F298) = 1;
  LODWORD(qword_4F8F298) = 100;
  qword_4F8F230 = 91;
  LOBYTE(dword_4F8F20C) = dword_4F8F20C & 0x9F | 0x20;
  qword_4F8F228 = (__int64)"The number of instructions to scan in a block in memory dependency analysis (default = 100)";
  sub_C53130(&qword_4F8F200);
  __cxa_atexit(sub_984970, &qword_4F8F200, &qword_4A427C0);
  qword_4F8F120 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8F170 = 0x100000000LL;
  dword_4F8F12C &= 0x8000u;
  word_4F8F130 = 0;
  qword_4F8F138 = 0;
  qword_4F8F140 = 0;
  dword_4F8F128 = v4;
  qword_4F8F148 = 0;
  qword_4F8F150 = 0;
  qword_4F8F158 = 0;
  qword_4F8F160 = 0;
  qword_4F8F168 = (__int64)&unk_4F8F178;
  qword_4F8F180 = 0;
  qword_4F8F188 = (__int64)&unk_4F8F1A0;
  qword_4F8F190 = 1;
  dword_4F8F198 = 0;
  byte_4F8F19C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F8F170;
  if ( (unsigned __int64)(unsigned int)qword_4F8F170 + 1 > HIDWORD(qword_4F8F170) )
  {
    v19 = v5;
    sub_C8D5F0((char *)&unk_4F8F178 - 16, &unk_4F8F178, (unsigned int)qword_4F8F170 + 1LL, 8);
    v6 = (unsigned int)qword_4F8F170;
    v5 = v19;
  }
  *(_QWORD *)(qword_4F8F168 + 8 * v6) = v5;
  qword_4F8F1B0 = (__int64)&unk_49D9728;
  qword_4F8F120 = (__int64)&unk_49DBF10;
  LODWORD(qword_4F8F170) = qword_4F8F170 + 1;
  qword_4F8F1A8 = 0;
  qword_4F8F1C0 = (__int64)&unk_49DC290;
  qword_4F8F1B8 = 0;
  qword_4F8F1E0 = (__int64)nullsub_24;
  qword_4F8F1D8 = (__int64)sub_984050;
  sub_C53080(&qword_4F8F120, "memdep-block-number-limit", 25);
  LODWORD(qword_4F8F1A8) = 200;
  BYTE4(qword_4F8F1B8) = 1;
  LODWORD(qword_4F8F1B8) = 200;
  qword_4F8F150 = 78;
  LOBYTE(dword_4F8F12C) = dword_4F8F12C & 0x9F | 0x20;
  qword_4F8F148 = (__int64)"The number of blocks to scan during memory dependency analysis (default = 200)";
  sub_C53130(&qword_4F8F120);
  __cxa_atexit(sub_984970, &qword_4F8F120, &qword_4A427C0);
  qword_4F8F040 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8F090 = 0x100000000LL;
  dword_4F8F04C &= 0x8000u;
  word_4F8F050 = 0;
  qword_4F8F058 = 0;
  qword_4F8F060 = 0;
  dword_4F8F048 = v7;
  qword_4F8F068 = 0;
  qword_4F8F070 = 0;
  qword_4F8F078 = 0;
  qword_4F8F080 = 0;
  qword_4F8F088 = (__int64)&unk_4F8F098;
  qword_4F8F0A0 = 0;
  qword_4F8F0A8 = (__int64)&unk_4F8F0C0;
  qword_4F8F0B0 = 1;
  dword_4F8F0B8 = 0;
  byte_4F8F0BC = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4F8F090;
  v10 = (unsigned int)qword_4F8F090 + 1LL;
  if ( v10 > HIDWORD(qword_4F8F090) )
  {
    sub_C8D5F0((char *)&unk_4F8F098 - 16, &unk_4F8F098, v10, 8);
    v9 = (unsigned int)qword_4F8F090;
  }
  *(_QWORD *)(qword_4F8F088 + 8 * v9) = v8;
  qword_4F8F0D0 = (__int64)&unk_49D9748;
  qword_4F8F040 = (__int64)&unk_49DC090;
  qword_4F8F0E0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F8F090) = qword_4F8F090 + 1;
  qword_4F8F100 = (__int64)nullsub_23;
  qword_4F8F0C8 = 0;
  qword_4F8F0F8 = (__int64)sub_984030;
  qword_4F8F0D8 = 0;
  sub_C53080(&qword_4F8F040, "memdep-cache-byval-loads", 24);
  LOWORD(qword_4F8F0D8) = 257;
  LOBYTE(qword_4F8F0C8) = 1;
  qword_4F8F070 = 61;
  LOBYTE(dword_4F8F04C) = dword_4F8F04C & 0x9F | 0x20;
  qword_4F8F068 = (__int64)"Preprocess byval loads to reduce compile-time  (default=true)";
  sub_C53130(&qword_4F8F040);
  __cxa_atexit(sub_984900, &qword_4F8F040, &qword_4A427C0);
  qword_4F8EF60 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8EFB0 = 0x100000000LL;
  dword_4F8EF6C &= 0x8000u;
  qword_4F8EFA8 = (__int64)&unk_4F8EFB8;
  word_4F8EF70 = 0;
  qword_4F8EF78 = 0;
  dword_4F8EF68 = v11;
  qword_4F8EF80 = 0;
  qword_4F8EF88 = 0;
  qword_4F8EF90 = 0;
  qword_4F8EF98 = 0;
  qword_4F8EFA0 = 0;
  qword_4F8EFC0 = 0;
  qword_4F8EFC8 = (__int64)&unk_4F8EFE0;
  qword_4F8EFD0 = 1;
  dword_4F8EFD8 = 0;
  byte_4F8EFDC = 1;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_4F8EFB0;
  if ( (unsigned __int64)(unsigned int)qword_4F8EFB0 + 1 > HIDWORD(qword_4F8EFB0) )
  {
    v20 = v12;
    sub_C8D5F0((char *)&unk_4F8EFB8 - 16, &unk_4F8EFB8, (unsigned int)qword_4F8EFB0 + 1LL, 8);
    v13 = (unsigned int)qword_4F8EFB0;
    v12 = v20;
  }
  *(_QWORD *)(qword_4F8EFA8 + 8 * v13) = v12;
  qword_4F8EFF0 = (__int64)&unk_49D9748;
  qword_4F8EF60 = (__int64)&unk_49DC090;
  qword_4F8F000 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F8EFB0) = qword_4F8EFB0 + 1;
  qword_4F8F020 = (__int64)nullsub_23;
  qword_4F8EFE8 = 0;
  qword_4F8F018 = (__int64)sub_984030;
  qword_4F8EFF8 = 0;
  sub_C53080(&qword_4F8EF60, "memdep-cache-candidates", 23);
  LOWORD(qword_4F8EFF8) = 257;
  LOBYTE(qword_4F8EFE8) = 1;
  qword_4F8EF90 = 72;
  LOBYTE(dword_4F8EF6C) = dword_4F8EF6C & 0x9F | 0x20;
  qword_4F8EF88 = (__int64)"Cache memory dependency candidates to reduce compile time (default=true)";
  sub_C53130(&qword_4F8EF60);
  __cxa_atexit(sub_984900, &qword_4F8EF60, &qword_4A427C0);
  qword_4F8EE80 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8EED0 = 0x100000000LL;
  dword_4F8EE8C &= 0x8000u;
  word_4F8EE90 = 0;
  qword_4F8EEC8 = (__int64)&unk_4F8EED8;
  qword_4F8EE98 = 0;
  dword_4F8EE88 = v14;
  qword_4F8EEA0 = 0;
  qword_4F8EEA8 = 0;
  qword_4F8EEB0 = 0;
  qword_4F8EEB8 = 0;
  qword_4F8EEC0 = 0;
  qword_4F8EEE0 = 0;
  qword_4F8EEE8 = (__int64)&unk_4F8EF00;
  qword_4F8EEF0 = 1;
  dword_4F8EEF8 = 0;
  byte_4F8EEFC = 1;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_4F8EED0;
  v17 = (unsigned int)qword_4F8EED0 + 1LL;
  if ( v17 > HIDWORD(qword_4F8EED0) )
  {
    sub_C8D5F0((char *)&unk_4F8EED8 - 16, &unk_4F8EED8, v17, 8);
    v16 = (unsigned int)qword_4F8EED0;
  }
  *(_QWORD *)(qword_4F8EEC8 + 8 * v16) = v15;
  qword_4F8EF10 = (__int64)&unk_49D9748;
  qword_4F8EE80 = (__int64)&unk_49DC090;
  qword_4F8EF20 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F8EED0) = qword_4F8EED0 + 1;
  qword_4F8EF40 = (__int64)nullsub_23;
  qword_4F8EF08 = 0;
  qword_4F8EF38 = (__int64)sub_984030;
  qword_4F8EF18 = 0;
  sub_C53080(&qword_4F8EE80, "memdep-cache-candidates-verify", 30);
  LOBYTE(qword_4F8EF08) = 0;
  qword_4F8EEB0 = 56;
  LOBYTE(dword_4F8EE8C) = dword_4F8EE8C & 0x9F | 0x20;
  LOWORD(qword_4F8EF18) = 256;
  qword_4F8EEA8 = (__int64)"[DebugOnly] Verify correctness of memdep candidate cache";
  sub_C53130(&qword_4F8EE80);
  return __cxa_atexit(sub_984900, &qword_4F8EE80, &qword_4A427C0);
}

// Function: ctor_516_0
// Address: 0x5605f0
//
int ctor_516_0()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r9
  __int64 v10; // r9
  __int64 v11; // r9
  int v12; // edx
  __int64 v13; // r13
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  int v16; // edx
  __int64 v17; // rbx
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v21; // [rsp+0h] [rbp-70h] BYREF
  __int64 *v22; // [rsp+8h] [rbp-68h] BYREF
  const char *v23; // [rsp+10h] [rbp-60h] BYREF
  __int64 v24; // [rsp+18h] [rbp-58h]
  char v25; // [rsp+30h] [rbp-40h]
  char v26; // [rsp+31h] [rbp-3Fh]

  qword_500F4C0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500F510 = 0x100000000LL;
  dword_500F4CC &= 0x8000u;
  word_500F4D0 = 0;
  qword_500F4D8 = 0;
  qword_500F4E0 = 0;
  dword_500F4C8 = v0;
  qword_500F4E8 = 0;
  qword_500F4F0 = 0;
  qword_500F4F8 = 0;
  qword_500F500 = 0;
  qword_500F508 = (__int64)&unk_500F518;
  qword_500F520 = 0;
  qword_500F528 = (__int64)&unk_500F540;
  qword_500F530 = 1;
  dword_500F538 = 0;
  byte_500F53C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_500F510;
  v3 = (unsigned int)qword_500F510 + 1LL;
  if ( v3 > HIDWORD(qword_500F510) )
  {
    sub_C8D5F0((char *)&unk_500F518 - 16, &unk_500F518, v3, 8);
    v2 = (unsigned int)qword_500F510;
  }
  *(_QWORD *)(qword_500F508 + 8 * v2) = v1;
  LODWORD(qword_500F510) = qword_500F510 + 1;
  qword_500F548 = 0;
  qword_500F550 = (__int64)&unk_49D9728;
  qword_500F558 = 0;
  qword_500F4C0 = (__int64)&unk_49DBF10;
  qword_500F560 = (__int64)&unk_49DC290;
  qword_500F580 = (__int64)nullsub_24;
  qword_500F578 = (__int64)sub_984050;
  sub_C53080(&qword_500F4C0, "lsv-max-chain-size", 18);
  qword_500F4F0 = 42;
  LODWORD(qword_500F548) = 1024;
  BYTE4(qword_500F558) = 1;
  LODWORD(qword_500F558) = 1024;
  LOBYTE(dword_500F4CC) = dword_500F4CC & 0x9F | 0x20;
  qword_500F4E8 = (__int64)"Maximum number of load/stores to vectorize";
  sub_C53130(&qword_500F4C0);
  __cxa_atexit(sub_984970, &qword_500F4C0, &qword_4A427C0);
  qword_500F420 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  word_500F430 = 0;
  qword_500F438 = 0;
  qword_500F440 = 0;
  dword_500F42C = dword_500F42C & 0x8000 | 0x20;
  qword_500F470 = 0x100000000LL;
  dword_500F428 = v4;
  qword_500F448 = 0;
  qword_500F450 = 0;
  qword_500F458 = 0;
  qword_500F460 = 0;
  qword_500F468 = (__int64)&unk_500F478;
  qword_500F480 = 0;
  qword_500F488 = (__int64)&unk_500F4A0;
  qword_500F490 = 1;
  dword_500F498 = 0;
  byte_500F49C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_500F470;
  v7 = (unsigned int)qword_500F470 + 1LL;
  if ( v7 > HIDWORD(qword_500F470) )
  {
    sub_C8D5F0((char *)&unk_500F478 - 16, &unk_500F478, v7, 8);
    v6 = (unsigned int)qword_500F470;
  }
  *(_QWORD *)(qword_500F468 + 8 * v6) = v5;
  LODWORD(qword_500F470) = qword_500F470 + 1;
  qword_500F4A8 = 0;
  qword_500F420 = (__int64)&unk_49DC380;
  sub_C53080(&qword_500F420, "max-chain-size", 14);
  qword_500F450 = 29;
  qword_500F448 = (__int64)"Alias for -lsv-max-chain-size";
  if ( qword_500F4A8 )
  {
    v8 = sub_CEADF0();
    v26 = 1;
    v23 = "cl::alias must only have one cl::aliasopt(...) specified!";
    v25 = 3;
    sub_C53280(&qword_500F420, &v23, 0, 0, v8);
  }
  qword_500F4A8 = (__int64)&qword_500F4C0;
  sub_C53EE0(&qword_500F420);
  __cxa_atexit(sub_C4FC50, &qword_500F420, &qword_4A427C0);
  LOBYTE(v21) = 0;
  v23 = "Should longer sequences of small datatypes be considered for upsizing during vectorization.";
  v22 = &v21;
  v24 = 91;
  HIDWORD(v21) = 1;
  ((void (__fastcall *)(void *, const char *, char *, const char **, __int64 **, __int64, __int64))sub_2AFC8D0)(
    &unk_500F340,
    "disable-ldst-upsizing",
    (char *)&v21 + 4,
    &v23,
    &v22,
    v9,
    v21);
  __cxa_atexit(sub_984900, &unk_500F340, &qword_4A427C0);
  v22 = (__int64 *)((char *)&v21 + 4);
  v23 = "Skip vectorization above this bb count";
  v24 = 38;
  v21 = 1;
  ((void (__fastcall *)(void *, const char *, __int64 *, const char **, __int64 **))sub_2AFCAE0)(
    &unk_500F260,
    "skip-vec-bb-ge",
    &v21,
    &v23,
    &v22);
  __cxa_atexit(sub_984970, &unk_500F260, &qword_4A427C0);
  v24 = 19;
  v22 = (__int64 *)byte_3F871B3;
  v23 = "Vectorize Only Func";
  HIDWORD(v21) = 1;
  sub_2AFCD00(&unk_500F160, "vec-only-func", (char *)&v21 + 4, &v23, &v22);
  __cxa_atexit(sub_BC5A40, &unk_500F160, &qword_4A427C0);
  LOBYTE(v21) = 1;
  v23 = "Should sequences of smaller datatypes in an aggregate be merged to a wider datatype before vectorization.";
  v22 = &v21;
  v24 = 105;
  HIDWORD(v21) = 1;
  ((void (__fastcall *)(void *, const char *, char *, const char **, __int64 **, __int64, __int64))sub_2AFC8D0)(
    &unk_500F080,
    "vect-split-aggr-merge",
    (char *)&v21 + 4,
    &v23,
    &v22,
    v10,
    v21);
  __cxa_atexit(sub_984900, &unk_500F080, &qword_4A427C0);
  v22 = (__int64 *)((char *)&v21 + 4);
  v23 = "Aggregates containing large number of elements will not be split";
  v24 = 64;
  v21 = 0x3200000001LL;
  ((void (__fastcall *)(void *, const char *, __int64 *, const char **, __int64 **, __int64))sub_2AFCAE0)(
    &unk_500EFA0,
    "max-aggr-elems",
    &v21,
    &v23,
    &v22,
    v11);
  __cxa_atexit(sub_984970, &unk_500EFA0, &qword_4A427C0);
  qword_500EEC0 = (__int64)&unk_49DC150;
  v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500EF10 = 0x100000000LL;
  dword_500EECC &= 0x8000u;
  word_500EED0 = 0;
  qword_500EED8 = 0;
  qword_500EEE0 = 0;
  dword_500EEC8 = v12;
  qword_500EEE8 = 0;
  qword_500EEF0 = 0;
  qword_500EEF8 = 0;
  qword_500EF00 = 0;
  qword_500EF08 = (__int64)&unk_500EF18;
  qword_500EF20 = 0;
  qword_500EF28 = (__int64)&unk_500EF40;
  qword_500EF30 = 1;
  dword_500EF38 = 0;
  byte_500EF3C = 1;
  v13 = sub_C57470();
  v14 = (unsigned int)qword_500EF10;
  v15 = (unsigned int)qword_500EF10 + 1LL;
  if ( v15 > HIDWORD(qword_500EF10) )
  {
    sub_C8D5F0((char *)&unk_500EF18 - 16, &unk_500EF18, v15, 8);
    v14 = (unsigned int)qword_500EF10;
  }
  *(_QWORD *)(qword_500EF08 + 8 * v14) = v13;
  qword_500EF50 = (__int64)&unk_49D9748;
  qword_500EEC0 = (__int64)&unk_49DC090;
  LODWORD(qword_500EF10) = qword_500EF10 + 1;
  qword_500EF48 = 0;
  qword_500EF60 = (__int64)&unk_49DC1D0;
  qword_500EF58 = 0;
  qword_500EF80 = (__int64)nullsub_23;
  qword_500EF78 = (__int64)sub_984030;
  sub_C53080(&qword_500EEC0, "vect-split-aggr", 15);
  LOWORD(qword_500EF58) = 257;
  LOBYTE(qword_500EF48) = 1;
  qword_500EEF0 = 48;
  LOBYTE(dword_500EECC) = dword_500EECC & 0x9F | 0x20;
  qword_500EEE8 = (__int64)"Should aggregates be split before vectorization.";
  sub_C53130(&qword_500EEC0);
  __cxa_atexit(sub_984900, &qword_500EEC0, &qword_4A427C0);
  qword_500EDE0 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500EE30 = 0x100000000LL;
  word_500EDF0 = 0;
  dword_500EDEC &= 0x8000u;
  qword_500EDF8 = 0;
  qword_500EE00 = 0;
  dword_500EDE8 = v16;
  qword_500EE08 = 0;
  qword_500EE10 = 0;
  qword_500EE18 = 0;
  qword_500EE20 = 0;
  qword_500EE28 = (__int64)&unk_500EE38;
  qword_500EE40 = 0;
  qword_500EE48 = (__int64)&unk_500EE60;
  qword_500EE50 = 1;
  dword_500EE58 = 0;
  byte_500EE5C = 1;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_500EE30;
  v19 = (unsigned int)qword_500EE30 + 1LL;
  if ( v19 > HIDWORD(qword_500EE30) )
  {
    sub_C8D5F0((char *)&unk_500EE38 - 16, &unk_500EE38, v19, 8);
    v18 = (unsigned int)qword_500EE30;
  }
  *(_QWORD *)(qword_500EE28 + 8 * v18) = v17;
  qword_500EE70 = (__int64)&unk_49D9748;
  qword_500EDE0 = (__int64)&unk_49DC090;
  LODWORD(qword_500EE30) = qword_500EE30 + 1;
  qword_500EE68 = 0;
  qword_500EE80 = (__int64)&unk_49DC1D0;
  qword_500EE78 = 0;
  qword_500EEA0 = (__int64)nullsub_23;
  qword_500EE98 = (__int64)sub_984030;
  sub_C53080(&qword_500EDE0, "aggressive-lsv", 14);
  qword_500EE10 = 64;
  LOBYTE(qword_500EE68) = 0;
  LOBYTE(dword_500EDEC) = dword_500EDEC & 0x9F | 0x20;
  qword_500EE08 = (__int64)"Allow expensive analysis for aggressive load-store vectorization";
  LOWORD(qword_500EE78) = 256;
  sub_C53130(&qword_500EDE0);
  return __cxa_atexit(sub_984900, &qword_500EDE0, &qword_4A427C0);
}

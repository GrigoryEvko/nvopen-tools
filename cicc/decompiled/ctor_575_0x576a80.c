// Function: ctor_575
// Address: 0x576a80
//
int ctor_575()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5022540 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5022590 = 0x100000000LL;
  word_5022550 = 0;
  dword_502254C &= 0x8000u;
  qword_5022558 = 0;
  qword_5022560 = 0;
  dword_5022548 = v0;
  qword_5022568 = 0;
  qword_5022570 = 0;
  qword_5022578 = 0;
  qword_5022580 = 0;
  qword_5022588 = (__int64)&unk_5022598;
  qword_50225A0 = 0;
  qword_50225A8 = (__int64)&unk_50225C0;
  qword_50225B0 = 1;
  dword_50225B8 = 0;
  byte_50225BC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5022590;
  v3 = (unsigned int)qword_5022590 + 1LL;
  if ( v3 > HIDWORD(qword_5022590) )
  {
    sub_C8D5F0((char *)&unk_5022598 - 16, &unk_5022598, v3, 8);
    v2 = (unsigned int)qword_5022590;
  }
  *(_QWORD *)(qword_5022588 + 8 * v2) = v1;
  qword_50225D0 = (__int64)&unk_49D9748;
  LODWORD(qword_5022590) = qword_5022590 + 1;
  qword_50225C8 = 0;
  qword_5022540 = (__int64)&unk_49DC090;
  qword_50225E0 = (__int64)&unk_49DC1D0;
  qword_50225D8 = 0;
  qword_5022600 = (__int64)nullsub_23;
  qword_50225F8 = (__int64)sub_984030;
  sub_C53080(&qword_5022540, "simplify-mir", 12);
  qword_5022570 = 51;
  LOBYTE(dword_502254C) = dword_502254C & 0x9F | 0x20;
  qword_5022568 = (__int64)"Leave out unnecessary information when printing MIR";
  sub_C53130(&qword_5022540);
  __cxa_atexit(sub_984900, &qword_5022540, &qword_4A427C0);
  qword_5022460 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_50224DC = 1;
  qword_50224B0 = 0x100000000LL;
  dword_502246C &= 0x8000u;
  qword_5022478 = 0;
  qword_5022480 = 0;
  qword_5022488 = 0;
  dword_5022468 = v4;
  word_5022470 = 0;
  qword_5022490 = 0;
  qword_5022498 = 0;
  qword_50224A0 = 0;
  qword_50224A8 = (__int64)&unk_50224B8;
  qword_50224C0 = 0;
  qword_50224C8 = (__int64)&unk_50224E0;
  qword_50224D0 = 1;
  dword_50224D8 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50224B0;
  v7 = (unsigned int)qword_50224B0 + 1LL;
  if ( v7 > HIDWORD(qword_50224B0) )
  {
    sub_C8D5F0((char *)&unk_50224B8 - 16, &unk_50224B8, v7, 8);
    v6 = (unsigned int)qword_50224B0;
  }
  *(_QWORD *)(qword_50224A8 + 8 * v6) = v5;
  qword_50224F0 = (__int64)&unk_49D9748;
  LODWORD(qword_50224B0) = qword_50224B0 + 1;
  qword_50224E8 = 0;
  qword_5022460 = (__int64)&unk_49DC090;
  qword_5022500 = (__int64)&unk_49DC1D0;
  qword_50224F8 = 0;
  qword_5022520 = (__int64)nullsub_23;
  qword_5022518 = (__int64)sub_984030;
  sub_C53080(&qword_5022460, "mir-debug-loc", 13);
  LOBYTE(qword_50224E8) = 1;
  qword_5022490 = 25;
  LOBYTE(dword_502246C) = dword_502246C & 0x9F | 0x20;
  LOWORD(qword_50224F8) = 257;
  qword_5022488 = (__int64)"Print MIR debug-locations";
  sub_C53130(&qword_5022460);
  return __cxa_atexit(sub_984900, &qword_5022460, &qword_4A427C0);
}

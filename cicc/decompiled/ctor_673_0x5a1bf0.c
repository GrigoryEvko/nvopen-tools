// Function: ctor_673
// Address: 0x5a1bf0
//
int __fastcall ctor_673(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
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
  __int64 v17; // rbx
  __int64 v18; // rax
  unsigned __int64 v19; // rdx

  qword_503D420 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_503D470 = 0x100000000LL;
  dword_503D42C &= 0x8000u;
  word_503D430 = 0;
  qword_503D438 = 0;
  qword_503D440 = 0;
  dword_503D428 = v4;
  qword_503D448 = 0;
  qword_503D450 = 0;
  qword_503D458 = 0;
  qword_503D460 = 0;
  qword_503D468 = (__int64)&unk_503D478;
  qword_503D480 = 0;
  qword_503D488 = (__int64)&unk_503D4A0;
  qword_503D490 = 1;
  dword_503D498 = 0;
  byte_503D49C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_503D470;
  v7 = (unsigned int)qword_503D470 + 1LL;
  if ( v7 > HIDWORD(qword_503D470) )
  {
    sub_C8D5F0((char *)&unk_503D478 - 16, &unk_503D478, v7, 8);
    v6 = (unsigned int)qword_503D470;
  }
  *(_QWORD *)(qword_503D468 + 8 * v6) = v5;
  LODWORD(qword_503D470) = qword_503D470 + 1;
  qword_503D4A8 = 0;
  qword_503D4B0 = (__int64)&unk_49D9728;
  qword_503D4B8 = 0;
  qword_503D420 = (__int64)&unk_49DBF10;
  qword_503D4C0 = (__int64)&unk_49DC290;
  qword_503D4E0 = (__int64)nullsub_24;
  qword_503D4D8 = (__int64)sub_984050;
  sub_C53080(&qword_503D420, "machine-combiner-inc-threshold", 30);
  qword_503D450 = 83;
  LODWORD(qword_503D4A8) = 500;
  BYTE4(qword_503D4B8) = 1;
  LODWORD(qword_503D4B8) = 500;
  LOBYTE(dword_503D42C) = dword_503D42C & 0x9F | 0x20;
  qword_503D448 = (__int64)"Incremental depth computation will be used for basic blocks with more instructions.";
  sub_C53130(&qword_503D420);
  __cxa_atexit(sub_984970, &qword_503D420, &qword_4A427C0);
  qword_503D340 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_503D420, v8, v9), 1u);
  qword_503D390 = 0x100000000LL;
  dword_503D34C &= 0x8000u;
  word_503D350 = 0;
  qword_503D358 = 0;
  qword_503D360 = 0;
  dword_503D348 = v10;
  qword_503D368 = 0;
  qword_503D370 = 0;
  qword_503D378 = 0;
  qword_503D380 = 0;
  qword_503D388 = (__int64)&unk_503D398;
  qword_503D3A0 = 0;
  qword_503D3A8 = (__int64)&unk_503D3C0;
  qword_503D3B0 = 1;
  dword_503D3B8 = 0;
  byte_503D3BC = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_503D390;
  v13 = (unsigned int)qword_503D390 + 1LL;
  if ( v13 > HIDWORD(qword_503D390) )
  {
    sub_C8D5F0((char *)&unk_503D398 - 16, &unk_503D398, v13, 8);
    v12 = (unsigned int)qword_503D390;
  }
  *(_QWORD *)(qword_503D388 + 8 * v12) = v11;
  qword_503D3D0 = (__int64)&unk_49D9748;
  qword_503D340 = (__int64)&unk_49DC090;
  LODWORD(qword_503D390) = qword_503D390 + 1;
  qword_503D3C8 = 0;
  qword_503D3E0 = (__int64)&unk_49DC1D0;
  qword_503D3D8 = 0;
  qword_503D400 = (__int64)nullsub_23;
  qword_503D3F8 = (__int64)sub_984030;
  sub_C53080(&qword_503D340, "machine-combiner-dump-subst-intrs", 33);
  LOWORD(qword_503D3D8) = 256;
  LOBYTE(qword_503D3C8) = 0;
  qword_503D370 = 26;
  LOBYTE(dword_503D34C) = dword_503D34C & 0x9F | 0x20;
  qword_503D368 = (__int64)"Dump all substituted intrs";
  sub_C53130(&qword_503D340);
  __cxa_atexit(sub_984900, &qword_503D340, &qword_4A427C0);
  qword_503D260 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_503D340, v14, v15), 1u);
  qword_503D2B0 = 0x100000000LL;
  word_503D270 = 0;
  dword_503D26C &= 0x8000u;
  qword_503D278 = 0;
  qword_503D280 = 0;
  dword_503D268 = v16;
  qword_503D288 = 0;
  qword_503D290 = 0;
  qword_503D298 = 0;
  qword_503D2A0 = 0;
  qword_503D2A8 = (__int64)&unk_503D2B8;
  qword_503D2C0 = 0;
  qword_503D2C8 = (__int64)&unk_503D2E0;
  qword_503D2D0 = 1;
  dword_503D2D8 = 0;
  byte_503D2DC = 1;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_503D2B0;
  v19 = (unsigned int)qword_503D2B0 + 1LL;
  if ( v19 > HIDWORD(qword_503D2B0) )
  {
    sub_C8D5F0((char *)&unk_503D2B8 - 16, &unk_503D2B8, v19, 8);
    v18 = (unsigned int)qword_503D2B0;
  }
  *(_QWORD *)(qword_503D2A8 + 8 * v18) = v17;
  qword_503D2F0 = (__int64)&unk_49D9748;
  qword_503D260 = (__int64)&unk_49DC090;
  LODWORD(qword_503D2B0) = qword_503D2B0 + 1;
  qword_503D2E8 = 0;
  qword_503D300 = (__int64)&unk_49DC1D0;
  qword_503D2F8 = 0;
  qword_503D320 = (__int64)nullsub_23;
  qword_503D318 = (__int64)sub_984030;
  sub_C53080(&qword_503D260, "machine-combiner-verify-pattern-order", 37);
  qword_503D290 = 68;
  LOBYTE(qword_503D2E8) = 0;
  LOBYTE(dword_503D26C) = dword_503D26C & 0x9F | 0x20;
  qword_503D288 = (__int64)"Verify that the generated patterns are ordered by increasing latency";
  LOWORD(qword_503D2F8) = 256;
  sub_C53130(&qword_503D260);
  return __cxa_atexit(sub_984900, &qword_503D260, &qword_4A427C0);
}

// Function: ctor_488
// Address: 0x553490
//
__int64 ctor_488()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // rbx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 result; // rax
  __int64 v21; // [rsp+8h] [rbp-78h]
  __int64 v22; // [rsp+8h] [rbp-78h]
  _QWORD v23[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v24[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v25[2]; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v26[8]; // [rsp+40h] [rbp-40h] BYREF

  qword_50076C0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5007710 = 0x100000000LL;
  dword_50076CC &= 0x8000u;
  word_50076D0 = 0;
  qword_50076D8 = 0;
  qword_50076E0 = 0;
  dword_50076C8 = v0;
  qword_50076E8 = 0;
  qword_50076F0 = 0;
  qword_50076F8 = 0;
  qword_5007700 = 0;
  qword_5007708 = (__int64)&unk_5007718;
  qword_5007720 = 0;
  qword_5007728 = (__int64)&unk_5007740;
  qword_5007730 = 1;
  dword_5007738 = 0;
  byte_500773C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5007710;
  v3 = (unsigned int)qword_5007710 + 1LL;
  if ( v3 > HIDWORD(qword_5007710) )
  {
    sub_C8D5F0((char *)&unk_5007718 - 16, &unk_5007718, v3, 8);
    v2 = (unsigned int)qword_5007710;
  }
  *(_QWORD *)(qword_5007708 + 8 * v2) = v1;
  qword_5007750 = (__int64)&unk_49D9748;
  qword_50076C0 = (__int64)&unk_49DC090;
  LODWORD(qword_5007710) = qword_5007710 + 1;
  qword_5007748 = 0;
  qword_5007760 = (__int64)&unk_49DC1D0;
  qword_5007758 = 0;
  qword_5007780 = (__int64)nullsub_23;
  qword_5007778 = (__int64)sub_984030;
  sub_C53080(&qword_50076C0, "enable-psr", 10);
  LOWORD(qword_5007758) = 257;
  LOBYTE(qword_5007748) = 1;
  qword_50076F0 = 30;
  LOBYTE(dword_50076CC) = dword_50076CC & 0x9F | 0x20;
  qword_50076E8 = (__int64)"Enable partial strength reduce";
  sub_C53130(&qword_50076C0);
  __cxa_atexit(sub_984900, &qword_50076C0, &qword_4A427C0);
  qword_50075E0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5007630 = 0x100000000LL;
  dword_50075EC &= 0x8000u;
  qword_5007628 = (__int64)&unk_5007638;
  word_50075F0 = 0;
  qword_50075F8 = 0;
  dword_50075E8 = v4;
  qword_5007600 = 0;
  qword_5007608 = 0;
  qword_5007610 = 0;
  qword_5007618 = 0;
  qword_5007620 = 0;
  qword_5007640 = 0;
  qword_5007648 = (__int64)&unk_5007660;
  qword_5007650 = 1;
  dword_5007658 = 0;
  byte_500765C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5007630;
  v7 = (unsigned int)qword_5007630 + 1LL;
  if ( v7 > HIDWORD(qword_5007630) )
  {
    sub_C8D5F0((char *)&unk_5007638 - 16, &unk_5007638, v7, 8);
    v6 = (unsigned int)qword_5007630;
  }
  *(_QWORD *)(qword_5007628 + 8 * v6) = v5;
  LODWORD(qword_5007630) = qword_5007630 + 1;
  qword_5007668 = 0;
  qword_5007670 = (__int64)&unk_49D9728;
  qword_5007678 = 0;
  qword_50075E0 = (__int64)&unk_49DBF10;
  qword_5007680 = (__int64)&unk_49DC290;
  qword_50076A0 = (__int64)nullsub_24;
  qword_5007698 = (__int64)sub_984050;
  sub_C53080(&qword_50075E0, "psr-threshold", 13);
  LODWORD(qword_5007668) = 4;
  BYTE4(qword_5007678) = 1;
  LODWORD(qword_5007678) = 4;
  qword_5007610 = 81;
  LOBYTE(dword_50075EC) = dword_50075EC & 0x9F | 0x20;
  qword_5007608 = (__int64)"The minimum number of candidates that's profitable to be optimized by a PSR basis";
  sub_C53130(&qword_50075E0);
  __cxa_atexit(sub_984970, &qword_50075E0, &qword_4A427C0);
  qword_5007500 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5007550 = 0x100000000LL;
  dword_500750C &= 0x8000u;
  qword_5007548 = (__int64)&unk_5007558;
  word_5007510 = 0;
  qword_5007518 = 0;
  dword_5007508 = v8;
  qword_5007520 = 0;
  qword_5007528 = 0;
  qword_5007530 = 0;
  qword_5007538 = 0;
  qword_5007540 = 0;
  qword_5007560 = 0;
  qword_5007568 = (__int64)&unk_5007580;
  qword_5007570 = 1;
  dword_5007578 = 0;
  byte_500757C = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_5007550;
  if ( (unsigned __int64)(unsigned int)qword_5007550 + 1 > HIDWORD(qword_5007550) )
  {
    v21 = v9;
    sub_C8D5F0((char *)&unk_5007558 - 16, &unk_5007558, (unsigned int)qword_5007550 + 1LL, 8);
    v10 = (unsigned int)qword_5007550;
    v9 = v21;
  }
  *(_QWORD *)(qword_5007548 + 8 * v10) = v9;
  qword_5007590 = (__int64)&unk_49D9748;
  qword_5007500 = (__int64)&unk_49DC090;
  LODWORD(qword_5007550) = qword_5007550 + 1;
  qword_5007588 = 0;
  qword_50075A0 = (__int64)&unk_49DC1D0;
  qword_5007598 = 0;
  qword_50075C0 = (__int64)nullsub_23;
  qword_50075B8 = (__int64)sub_984030;
  sub_C53080(&qword_5007500, "slsr-simplest-form", 18);
  LOWORD(qword_5007598) = 257;
  LOBYTE(qword_5007588) = 1;
  qword_5007530 = 37;
  LOBYTE(dword_500750C) = dword_500750C & 0x9F | 0x20;
  qword_5007528 = (__int64)"Process simplest form SLSR candidates";
  sub_C53130(&qword_5007500);
  __cxa_atexit(sub_984900, &qword_5007500, &qword_4A427C0);
  qword_5007420 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_500749C = 1;
  word_5007430 = 0;
  qword_5007470 = 0x100000000LL;
  dword_500742C &= 0x8000u;
  qword_5007468 = (__int64)&unk_5007478;
  qword_5007438 = 0;
  dword_5007428 = v11;
  qword_5007440 = 0;
  qword_5007448 = 0;
  qword_5007450 = 0;
  qword_5007458 = 0;
  qword_5007460 = 0;
  qword_5007480 = 0;
  qword_5007488 = (__int64)&unk_50074A0;
  qword_5007490 = 1;
  dword_5007498 = 0;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_5007470;
  if ( (unsigned __int64)(unsigned int)qword_5007470 + 1 > HIDWORD(qword_5007470) )
  {
    v22 = v12;
    sub_C8D5F0((char *)&unk_5007478 - 16, &unk_5007478, (unsigned int)qword_5007470 + 1LL, 8);
    v13 = (unsigned int)qword_5007470;
    v12 = v22;
  }
  *(_QWORD *)(qword_5007468 + 8 * v13) = v12;
  LODWORD(qword_5007470) = qword_5007470 + 1;
  qword_50074A8 = 0;
  qword_50074B0 = (__int64)&unk_49D9728;
  qword_50074B8 = 0;
  qword_5007420 = (__int64)&unk_49DBF10;
  qword_50074C0 = (__int64)&unk_49DC290;
  qword_50074E0 = (__int64)nullsub_24;
  qword_50074D8 = (__int64)sub_984050;
  sub_C53080(&qword_5007420, "max-psr-path-length", 19);
  LODWORD(qword_50074A8) = 200;
  BYTE4(qword_50074B8) = 1;
  LODWORD(qword_50074B8) = 200;
  qword_5007450 = 42;
  LOBYTE(dword_500742C) = dword_500742C & 0x9F | 0x20;
  qword_5007448 = (__int64)"The maximum number of blocks in a PSR path";
  sub_C53130(&qword_5007420);
  __cxa_atexit(sub_984970, &qword_5007420, &qword_4A427C0);
  qword_5007340 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_50073BC = 1;
  qword_5007390 = 0x100000000LL;
  dword_500734C &= 0x8000u;
  qword_5007358 = 0;
  qword_5007360 = 0;
  qword_5007368 = 0;
  dword_5007348 = v14;
  word_5007350 = 0;
  qword_5007370 = 0;
  qword_5007378 = 0;
  qword_5007380 = 0;
  qword_5007388 = (__int64)&unk_5007398;
  qword_50073A0 = 0;
  qword_50073A8 = (__int64)&unk_50073C0;
  qword_50073B0 = 1;
  dword_50073B8 = 0;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_5007390;
  v17 = (unsigned int)qword_5007390 + 1LL;
  if ( v17 > HIDWORD(qword_5007390) )
  {
    sub_C8D5F0((char *)&unk_5007398 - 16, &unk_5007398, v17, 8);
    v16 = (unsigned int)qword_5007390;
  }
  *(_QWORD *)(qword_5007388 + 8 * v16) = v15;
  qword_50073D0 = (__int64)&unk_49D9748;
  qword_5007340 = (__int64)&unk_49DC090;
  LODWORD(qword_5007390) = qword_5007390 + 1;
  qword_50073C8 = 0;
  qword_50073E0 = (__int64)&unk_49DC1D0;
  qword_50073D8 = 0;
  qword_5007400 = (__int64)nullsub_23;
  qword_50073F8 = (__int64)sub_984030;
  sub_C53080(&qword_5007340, "slsr-variable-delta-reuse", 25);
  LOBYTE(qword_50073C8) = 1;
  qword_5007370 = 46;
  LOBYTE(dword_500734C) = dword_500734C & 0x9F | 0x20;
  LOWORD(qword_50073D8) = 257;
  qword_5007368 = (__int64)"Reuse computation from variable delta for SLSR";
  sub_C53130(&qword_5007340);
  __cxa_atexit(sub_984900, &qword_5007340, &qword_4A427C0);
  v18 = sub_C60B10();
  v25[0] = v26;
  v19 = v18;
  sub_297BE00(v25, "Controls whether rewriteCandidateWithBasis is executed.");
  v23[0] = v24;
  sub_297BE00(v23, "slsr-counter");
  result = sub_CF9810(v19, v23, v25);
  if ( (_QWORD *)v23[0] != v24 )
    result = j_j___libc_free_0(v23[0], v24[0] + 1LL);
  if ( (_QWORD *)v25[0] != v26 )
    return j_j___libc_free_0(v25[0], v26[0] + 1LL);
  return result;
}

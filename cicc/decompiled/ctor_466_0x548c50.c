// Function: ctor_466
// Address: 0x548c50
//
int ctor_466()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rdx
  int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // edx
  __int64 v17; // rbx
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v21; // [rsp+8h] [rbp-38h]
  __int64 v22; // [rsp+8h] [rbp-38h]
  __int64 v23; // [rsp+8h] [rbp-38h]
  __int64 v24; // [rsp+8h] [rbp-38h]

  qword_50003A0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50003AC &= 0x8000u;
  word_50003B0 = 0;
  qword_50003F0 = 0x100000000LL;
  qword_50003B8 = 0;
  qword_50003C0 = 0;
  qword_50003C8 = 0;
  dword_50003A8 = v0;
  qword_50003D0 = 0;
  qword_50003D8 = 0;
  qword_50003E0 = 0;
  qword_50003E8 = (__int64)&unk_50003F8;
  qword_5000400 = 0;
  qword_5000408 = (__int64)&unk_5000420;
  qword_5000410 = 1;
  dword_5000418 = 0;
  byte_500041C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_50003F0;
  v3 = (unsigned int)qword_50003F0 + 1LL;
  if ( v3 > HIDWORD(qword_50003F0) )
  {
    sub_C8D5F0((char *)&unk_50003F8 - 16, &unk_50003F8, v3, 8);
    v2 = (unsigned int)qword_50003F0;
  }
  *(_QWORD *)(qword_50003E8 + 8 * v2) = v1;
  qword_5000430 = (__int64)&unk_49D9748;
  qword_50003A0 = (__int64)&unk_49DC090;
  qword_5000440 = (__int64)&unk_49DC1D0;
  LODWORD(qword_50003F0) = qword_50003F0 + 1;
  qword_5000460 = (__int64)nullsub_23;
  qword_5000428 = 0;
  qword_5000458 = (__int64)sub_984030;
  qword_5000438 = 0;
  sub_C53080(&qword_50003A0, "loop-predication-enable-iv-truncation", 37);
  LOWORD(qword_5000438) = 257;
  LOBYTE(qword_5000428) = 1;
  LOBYTE(dword_50003AC) = dword_50003AC & 0x9F | 0x20;
  sub_C53130(&qword_50003A0);
  __cxa_atexit(sub_984900, &qword_50003A0, &qword_4A427C0);
  qword_50002C0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50002CC &= 0x8000u;
  word_50002D0 = 0;
  qword_5000310 = 0x100000000LL;
  qword_5000308 = (__int64)&unk_5000318;
  qword_50002D8 = 0;
  qword_50002E0 = 0;
  dword_50002C8 = v4;
  qword_50002E8 = 0;
  qword_50002F0 = 0;
  qword_50002F8 = 0;
  qword_5000300 = 0;
  qword_5000320 = 0;
  qword_5000328 = (__int64)&unk_5000340;
  qword_5000330 = 1;
  dword_5000338 = 0;
  byte_500033C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5000310;
  if ( (unsigned __int64)(unsigned int)qword_5000310 + 1 > HIDWORD(qword_5000310) )
  {
    v21 = v5;
    sub_C8D5F0((char *)&unk_5000318 - 16, &unk_5000318, (unsigned int)qword_5000310 + 1LL, 8);
    v6 = (unsigned int)qword_5000310;
    v5 = v21;
  }
  *(_QWORD *)(qword_5000308 + 8 * v6) = v5;
  qword_5000350 = (__int64)&unk_49D9748;
  qword_50002C0 = (__int64)&unk_49DC090;
  qword_5000360 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5000310) = qword_5000310 + 1;
  qword_5000380 = (__int64)nullsub_23;
  qword_5000348 = 0;
  qword_5000378 = (__int64)sub_984030;
  qword_5000358 = 0;
  sub_C53080(&qword_50002C0, "loop-predication-enable-count-down-loop", 39);
  LOWORD(qword_5000358) = 257;
  LOBYTE(qword_5000348) = 1;
  LOBYTE(dword_50002CC) = dword_50002CC & 0x9F | 0x20;
  sub_C53130(&qword_50002C0);
  __cxa_atexit(sub_984900, &qword_50002C0, &qword_4A427C0);
  qword_50001E0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5000230 = 0x100000000LL;
  dword_50001EC &= 0x8000u;
  qword_5000228 = (__int64)&unk_5000238;
  word_50001F0 = 0;
  qword_50001F8 = 0;
  dword_50001E8 = v7;
  qword_5000200 = 0;
  qword_5000208 = 0;
  qword_5000210 = 0;
  qword_5000218 = 0;
  qword_5000220 = 0;
  qword_5000240 = 0;
  qword_5000248 = (__int64)&unk_5000260;
  qword_5000250 = 1;
  dword_5000258 = 0;
  byte_500025C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_5000230;
  if ( (unsigned __int64)(unsigned int)qword_5000230 + 1 > HIDWORD(qword_5000230) )
  {
    v22 = v8;
    sub_C8D5F0((char *)&unk_5000238 - 16, &unk_5000238, (unsigned int)qword_5000230 + 1LL, 8);
    v9 = (unsigned int)qword_5000230;
    v8 = v22;
  }
  *(_QWORD *)(qword_5000228 + 8 * v9) = v8;
  qword_5000270 = (__int64)&unk_49D9748;
  qword_50001E0 = (__int64)&unk_49DC090;
  qword_5000280 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5000230) = qword_5000230 + 1;
  qword_50002A0 = (__int64)nullsub_23;
  qword_5000268 = 0;
  qword_5000298 = (__int64)sub_984030;
  qword_5000278 = 0;
  sub_C53080(&qword_50001E0, "loop-predication-skip-profitability-checks", 42);
  LOBYTE(qword_5000268) = 0;
  LOWORD(qword_5000278) = 256;
  LOBYTE(dword_50001EC) = dword_50001EC & 0x9F | 0x20;
  sub_C53130(&qword_50001E0);
  __cxa_atexit(sub_984900, &qword_50001E0, &qword_4A427C0);
  qword_5000100 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5000150 = 0x100000000LL;
  dword_500010C &= 0x8000u;
  word_5000110 = 0;
  qword_5000148 = (__int64)&unk_5000158;
  qword_5000118 = 0;
  dword_5000108 = v10;
  qword_5000120 = 0;
  qword_5000128 = 0;
  qword_5000130 = 0;
  qword_5000138 = 0;
  qword_5000140 = 0;
  qword_5000160 = 0;
  qword_5000168 = (__int64)&unk_5000180;
  qword_5000170 = 1;
  dword_5000178 = 0;
  byte_500017C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5000150;
  if ( (unsigned __int64)(unsigned int)qword_5000150 + 1 > HIDWORD(qword_5000150) )
  {
    v23 = v11;
    sub_C8D5F0((char *)&unk_5000158 - 16, &unk_5000158, (unsigned int)qword_5000150 + 1LL, 8);
    v12 = (unsigned int)qword_5000150;
    v11 = v23;
  }
  *(_QWORD *)(qword_5000148 + 8 * v12) = v11;
  LODWORD(qword_5000150) = qword_5000150 + 1;
  qword_5000188 = 0;
  qword_5000190 = (__int64)&unk_49E5940;
  qword_5000198 = 0;
  qword_5000100 = (__int64)&unk_49E5960;
  qword_50001A0 = (__int64)&unk_49DC320;
  qword_50001C0 = (__int64)nullsub_385;
  qword_50001B8 = (__int64)sub_1038930;
  sub_C53080(&qword_5000100, "loop-predication-latch-probability-scale", 40);
  BYTE4(qword_5000198) = 1;
  LODWORD(qword_5000188) = 0x40000000;
  LODWORD(qword_5000198) = 0x40000000;
  LOBYTE(dword_500010C) = dword_500010C & 0x9F | 0x20;
  qword_5000128 = (__int64)"scale factor for the latch probability. Value should be greater than 1. Lower values are ignored";
  qword_5000130 = 96;
  sub_C53130(&qword_5000100);
  __cxa_atexit(sub_1038DB0, &qword_5000100, &qword_4A427C0);
  qword_5000020 = (__int64)&unk_49DC150;
  v13 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_500009C = 1;
  word_5000030 = 0;
  qword_5000070 = 0x100000000LL;
  dword_500002C &= 0x8000u;
  qword_5000068 = (__int64)&unk_5000078;
  qword_5000038 = 0;
  dword_5000028 = v13;
  qword_5000040 = 0;
  qword_5000048 = 0;
  qword_5000050 = 0;
  qword_5000058 = 0;
  qword_5000060 = 0;
  qword_5000080 = 0;
  qword_5000088 = (__int64)&unk_50000A0;
  qword_5000090 = 1;
  dword_5000098 = 0;
  v14 = sub_C57470();
  v15 = (unsigned int)qword_5000070;
  if ( (unsigned __int64)(unsigned int)qword_5000070 + 1 > HIDWORD(qword_5000070) )
  {
    v24 = v14;
    sub_C8D5F0((char *)&unk_5000078 - 16, &unk_5000078, (unsigned int)qword_5000070 + 1LL, 8);
    v15 = (unsigned int)qword_5000070;
    v14 = v24;
  }
  *(_QWORD *)(qword_5000068 + 8 * v15) = v14;
  qword_50000B0 = (__int64)&unk_49D9748;
  qword_5000020 = (__int64)&unk_49DC090;
  qword_50000C0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5000070) = qword_5000070 + 1;
  qword_50000E0 = (__int64)nullsub_23;
  qword_50000A8 = 0;
  qword_50000D8 = (__int64)sub_984030;
  qword_50000B8 = 0;
  sub_C53080(&qword_5000020, "loop-predication-predicate-widenable-branches-to-deopt", 54);
  LOWORD(qword_50000B8) = 257;
  LOBYTE(qword_50000A8) = 1;
  qword_5000050 = 94;
  LOBYTE(dword_500002C) = dword_500002C & 0x9F | 0x20;
  qword_5000048 = (__int64)"Whether or not we should predicate guards expressed as widenable branches to deoptimize blocks";
  sub_C53130(&qword_5000020);
  __cxa_atexit(sub_984900, &qword_5000020, &qword_4A427C0);
  qword_4FFFF40 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFFF90 = 0x100000000LL;
  dword_4FFFF4C &= 0x8000u;
  word_4FFFF50 = 0;
  qword_4FFFF88 = (__int64)&unk_4FFFF98;
  qword_4FFFF58 = 0;
  dword_4FFFF48 = v16;
  qword_4FFFF60 = 0;
  qword_4FFFF68 = 0;
  qword_4FFFF70 = 0;
  qword_4FFFF78 = 0;
  qword_4FFFF80 = 0;
  qword_4FFFFA0 = 0;
  qword_4FFFFA8 = (__int64)&unk_4FFFFC0;
  qword_4FFFFB0 = 1;
  dword_4FFFFB8 = 0;
  byte_4FFFFBC = 1;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_4FFFF90;
  v19 = (unsigned int)qword_4FFFF90 + 1LL;
  if ( v19 > HIDWORD(qword_4FFFF90) )
  {
    sub_C8D5F0((char *)&unk_4FFFF98 - 16, &unk_4FFFF98, v19, 8);
    v18 = (unsigned int)qword_4FFFF90;
  }
  *(_QWORD *)(qword_4FFFF88 + 8 * v18) = v17;
  qword_4FFFFD0 = (__int64)&unk_49D9748;
  qword_4FFFF40 = (__int64)&unk_49DC090;
  qword_4FFFFE0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FFFF90) = qword_4FFFF90 + 1;
  qword_5000000 = (__int64)nullsub_23;
  qword_4FFFFC8 = 0;
  qword_4FFFFF8 = (__int64)sub_984030;
  qword_4FFFFD8 = 0;
  sub_C53080(&qword_4FFFF40, "loop-predication-insert-assumes-of-predicated-guards-conditions", 63);
  qword_4FFFF70 = 74;
  LOBYTE(qword_4FFFFC8) = 1;
  LOBYTE(dword_4FFFF4C) = dword_4FFFF4C & 0x9F | 0x20;
  qword_4FFFF68 = (__int64)"Whether or not we should insert assumes of conditions of predicated guards";
  LOWORD(qword_4FFFFD8) = 257;
  sub_C53130(&qword_4FFFF40);
  return __cxa_atexit(sub_984900, &qword_4FFFF40, &qword_4A427C0);
}

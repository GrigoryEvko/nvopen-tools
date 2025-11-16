// Function: ctor_453
// Address: 0x5426c0
//
int ctor_453()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // edx
  __int64 v18; // rax
  __int64 v19; // rdx
  int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rdx
  int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rdx
  int v26; // edx
  __int64 v27; // rbx
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  __int64 v31; // [rsp+8h] [rbp-38h]
  __int64 v32; // [rsp+8h] [rbp-38h]
  __int64 v33; // [rsp+8h] [rbp-38h]
  __int64 v34; // [rsp+8h] [rbp-38h]
  __int64 v35; // [rsp+8h] [rbp-38h]
  __int64 v36; // [rsp+8h] [rbp-38h]

  qword_4FFD3C0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FFD43C = 1;
  qword_4FFD410 = 0x100000000LL;
  dword_4FFD3CC &= 0x8000u;
  qword_4FFD3D8 = 0;
  qword_4FFD3E0 = 0;
  qword_4FFD3E8 = 0;
  dword_4FFD3C8 = v0;
  word_4FFD3D0 = 0;
  qword_4FFD3F0 = 0;
  qword_4FFD3F8 = 0;
  qword_4FFD400 = 0;
  qword_4FFD408 = (__int64)&unk_4FFD418;
  qword_4FFD420 = 0;
  qword_4FFD428 = (__int64)&unk_4FFD440;
  qword_4FFD430 = 1;
  dword_4FFD438 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FFD410;
  v3 = (unsigned int)qword_4FFD410 + 1LL;
  if ( v3 > HIDWORD(qword_4FFD410) )
  {
    sub_C8D5F0((char *)&unk_4FFD418 - 16, &unk_4FFD418, v3, 8);
    v2 = (unsigned int)qword_4FFD410;
  }
  *(_QWORD *)(qword_4FFD408 + 8 * v2) = v1;
  LODWORD(qword_4FFD410) = qword_4FFD410 + 1;
  qword_4FFD448 = 0;
  qword_4FFD450 = (__int64)&unk_49D9728;
  qword_4FFD458 = 0;
  qword_4FFD3C0 = (__int64)&unk_49DBF10;
  qword_4FFD460 = (__int64)&unk_49DC290;
  qword_4FFD480 = (__int64)nullsub_24;
  qword_4FFD478 = (__int64)sub_984050;
  sub_C53080(&qword_4FFD3C0, "irce-loop-size-cutoff", 21);
  LODWORD(qword_4FFD448) = 64;
  BYTE4(qword_4FFD458) = 1;
  LODWORD(qword_4FFD458) = 64;
  LOBYTE(dword_4FFD3CC) = dword_4FFD3CC & 0x9F | 0x20;
  sub_C53130(&qword_4FFD3C0);
  __cxa_atexit(sub_984970, &qword_4FFD3C0, &qword_4A427C0);
  qword_4FFD2E0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FFD2EC &= 0x8000u;
  word_4FFD2F0 = 0;
  qword_4FFD330 = 0x100000000LL;
  qword_4FFD2F8 = 0;
  qword_4FFD300 = 0;
  qword_4FFD308 = 0;
  dword_4FFD2E8 = v4;
  qword_4FFD310 = 0;
  qword_4FFD318 = 0;
  qword_4FFD320 = 0;
  qword_4FFD328 = (__int64)&unk_4FFD338;
  qword_4FFD340 = 0;
  qword_4FFD348 = (__int64)&unk_4FFD360;
  qword_4FFD350 = 1;
  dword_4FFD358 = 0;
  byte_4FFD35C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FFD330;
  v7 = (unsigned int)qword_4FFD330 + 1LL;
  if ( v7 > HIDWORD(qword_4FFD330) )
  {
    sub_C8D5F0((char *)&unk_4FFD338 - 16, &unk_4FFD338, v7, 8);
    v6 = (unsigned int)qword_4FFD330;
  }
  *(_QWORD *)(qword_4FFD328 + 8 * v6) = v5;
  qword_4FFD370 = (__int64)&unk_49D9748;
  qword_4FFD2E0 = (__int64)&unk_49DC090;
  qword_4FFD380 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FFD330) = qword_4FFD330 + 1;
  qword_4FFD3A0 = (__int64)nullsub_23;
  qword_4FFD368 = 0;
  qword_4FFD398 = (__int64)sub_984030;
  qword_4FFD378 = 0;
  sub_C53080(&qword_4FFD2E0, "irce-print-changed-loops", 24);
  LOBYTE(qword_4FFD368) = 0;
  LOBYTE(dword_4FFD2EC) = dword_4FFD2EC & 0x9F | 0x20;
  LOWORD(qword_4FFD378) = 256;
  sub_C53130(&qword_4FFD2E0);
  __cxa_atexit(sub_984900, &qword_4FFD2E0, &qword_4A427C0);
  qword_4FFD200 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FFD20C &= 0x8000u;
  word_4FFD210 = 0;
  qword_4FFD250 = 0x100000000LL;
  qword_4FFD248 = (__int64)&unk_4FFD258;
  qword_4FFD218 = 0;
  qword_4FFD220 = 0;
  dword_4FFD208 = v8;
  qword_4FFD228 = 0;
  qword_4FFD230 = 0;
  qword_4FFD238 = 0;
  qword_4FFD240 = 0;
  qword_4FFD260 = 0;
  qword_4FFD268 = (__int64)&unk_4FFD280;
  qword_4FFD270 = 1;
  dword_4FFD278 = 0;
  byte_4FFD27C = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4FFD250;
  if ( (unsigned __int64)(unsigned int)qword_4FFD250 + 1 > HIDWORD(qword_4FFD250) )
  {
    v31 = v9;
    sub_C8D5F0((char *)&unk_4FFD258 - 16, &unk_4FFD258, (unsigned int)qword_4FFD250 + 1LL, 8);
    v10 = (unsigned int)qword_4FFD250;
    v9 = v31;
  }
  *(_QWORD *)(qword_4FFD248 + 8 * v10) = v9;
  qword_4FFD290 = (__int64)&unk_49D9748;
  qword_4FFD200 = (__int64)&unk_49DC090;
  qword_4FFD2A0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FFD250) = qword_4FFD250 + 1;
  qword_4FFD2C0 = (__int64)nullsub_23;
  qword_4FFD288 = 0;
  qword_4FFD2B8 = (__int64)sub_984030;
  qword_4FFD298 = 0;
  sub_C53080(&qword_4FFD200, "irce-print-range-checks", 23);
  LOBYTE(qword_4FFD288) = 0;
  LOBYTE(dword_4FFD20C) = dword_4FFD20C & 0x9F | 0x20;
  LOWORD(qword_4FFD298) = 256;
  sub_C53130(&qword_4FFD200);
  __cxa_atexit(sub_984900, &qword_4FFD200, &qword_4A427C0);
  qword_4FFD120 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FFD12C &= 0x8000u;
  word_4FFD130 = 0;
  qword_4FFD170 = 0x100000000LL;
  qword_4FFD168 = (__int64)&unk_4FFD178;
  qword_4FFD138 = 0;
  qword_4FFD140 = 0;
  dword_4FFD128 = v11;
  qword_4FFD148 = 0;
  qword_4FFD150 = 0;
  qword_4FFD158 = 0;
  qword_4FFD160 = 0;
  qword_4FFD180 = 0;
  qword_4FFD188 = (__int64)&unk_4FFD1A0;
  qword_4FFD190 = 1;
  dword_4FFD198 = 0;
  byte_4FFD19C = 1;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_4FFD170;
  if ( (unsigned __int64)(unsigned int)qword_4FFD170 + 1 > HIDWORD(qword_4FFD170) )
  {
    v32 = v12;
    sub_C8D5F0((char *)&unk_4FFD178 - 16, &unk_4FFD178, (unsigned int)qword_4FFD170 + 1LL, 8);
    v13 = (unsigned int)qword_4FFD170;
    v12 = v32;
  }
  *(_QWORD *)(qword_4FFD168 + 8 * v13) = v12;
  qword_4FFD1B0 = (__int64)&unk_49D9748;
  qword_4FFD120 = (__int64)&unk_49DC090;
  qword_4FFD1C0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FFD170) = qword_4FFD170 + 1;
  qword_4FFD1E0 = (__int64)nullsub_23;
  qword_4FFD1A8 = 0;
  qword_4FFD1D8 = (__int64)sub_984030;
  qword_4FFD1B8 = 0;
  sub_C53080(&qword_4FFD120, "irce-skip-profitability-checks", 30);
  LOWORD(qword_4FFD1B8) = 256;
  LOBYTE(qword_4FFD1A8) = 0;
  LOBYTE(dword_4FFD12C) = dword_4FFD12C & 0x9F | 0x20;
  sub_C53130(&qword_4FFD120);
  __cxa_atexit(sub_984900, &qword_4FFD120, &qword_4A427C0);
  qword_4FFD040 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFD090 = 0x100000000LL;
  dword_4FFD04C &= 0x8000u;
  qword_4FFD088 = (__int64)&unk_4FFD098;
  word_4FFD050 = 0;
  qword_4FFD058 = 0;
  dword_4FFD048 = v14;
  qword_4FFD060 = 0;
  qword_4FFD068 = 0;
  qword_4FFD070 = 0;
  qword_4FFD078 = 0;
  qword_4FFD080 = 0;
  qword_4FFD0A0 = 0;
  qword_4FFD0A8 = (__int64)&unk_4FFD0C0;
  qword_4FFD0B0 = 1;
  dword_4FFD0B8 = 0;
  byte_4FFD0BC = 1;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_4FFD090;
  if ( (unsigned __int64)(unsigned int)qword_4FFD090 + 1 > HIDWORD(qword_4FFD090) )
  {
    v33 = v15;
    sub_C8D5F0((char *)&unk_4FFD098 - 16, &unk_4FFD098, (unsigned int)qword_4FFD090 + 1LL, 8);
    v16 = (unsigned int)qword_4FFD090;
    v15 = v33;
  }
  *(_QWORD *)(qword_4FFD088 + 8 * v16) = v15;
  LODWORD(qword_4FFD090) = qword_4FFD090 + 1;
  qword_4FFD0C8 = 0;
  qword_4FFD0D0 = (__int64)&unk_49D9728;
  qword_4FFD0D8 = 0;
  qword_4FFD040 = (__int64)&unk_49DBF10;
  qword_4FFD0E0 = (__int64)&unk_49DC290;
  qword_4FFD100 = (__int64)nullsub_24;
  qword_4FFD0F8 = (__int64)sub_984050;
  sub_C53080(&qword_4FFD040, "irce-min-eliminated-checks", 26);
  LODWORD(qword_4FFD0C8) = 10;
  BYTE4(qword_4FFD0D8) = 1;
  LODWORD(qword_4FFD0D8) = 10;
  LOBYTE(dword_4FFD04C) = dword_4FFD04C & 0x9F | 0x20;
  sub_C53130(&qword_4FFD040);
  __cxa_atexit(sub_984970, &qword_4FFD040, &qword_4A427C0);
  qword_4FFCF60 = (__int64)&unk_49DC150;
  v17 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFCFB0 = 0x100000000LL;
  dword_4FFCF6C &= 0x8000u;
  qword_4FFCFA8 = (__int64)&unk_4FFCFB8;
  word_4FFCF70 = 0;
  qword_4FFCF78 = 0;
  dword_4FFCF68 = v17;
  qword_4FFCF80 = 0;
  qword_4FFCF88 = 0;
  qword_4FFCF90 = 0;
  qword_4FFCF98 = 0;
  qword_4FFCFA0 = 0;
  qword_4FFCFC0 = 0;
  qword_4FFCFC8 = (__int64)&unk_4FFCFE0;
  qword_4FFCFD0 = 1;
  dword_4FFCFD8 = 0;
  byte_4FFCFDC = 1;
  v18 = sub_C57470();
  v19 = (unsigned int)qword_4FFCFB0;
  if ( (unsigned __int64)(unsigned int)qword_4FFCFB0 + 1 > HIDWORD(qword_4FFCFB0) )
  {
    v34 = v18;
    sub_C8D5F0((char *)&unk_4FFCFB8 - 16, &unk_4FFCFB8, (unsigned int)qword_4FFCFB0 + 1LL, 8);
    v19 = (unsigned int)qword_4FFCFB0;
    v18 = v34;
  }
  *(_QWORD *)(qword_4FFCFA8 + 8 * v19) = v18;
  qword_4FFCFF0 = (__int64)&unk_49D9748;
  qword_4FFCF60 = (__int64)&unk_49DC090;
  qword_4FFD000 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FFCFB0) = qword_4FFCFB0 + 1;
  qword_4FFD020 = (__int64)nullsub_23;
  qword_4FFCFE8 = 0;
  qword_4FFD018 = (__int64)sub_984030;
  qword_4FFCFF8 = 0;
  sub_C53080(&qword_4FFCF60, "irce-allow-unsigned-latch", 25);
  LOBYTE(qword_4FFCFE8) = 1;
  LOWORD(qword_4FFCFF8) = 257;
  LOBYTE(dword_4FFCF6C) = dword_4FFCF6C & 0x9F | 0x20;
  sub_C53130(&qword_4FFCF60);
  __cxa_atexit(sub_984900, &qword_4FFCF60, &qword_4A427C0);
  qword_4FFCE80 = (__int64)&unk_49DC150;
  v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFCED0 = 0x100000000LL;
  dword_4FFCE8C &= 0x8000u;
  word_4FFCE90 = 0;
  qword_4FFCEC8 = (__int64)&unk_4FFCED8;
  qword_4FFCE98 = 0;
  dword_4FFCE88 = v20;
  qword_4FFCEA0 = 0;
  qword_4FFCEA8 = 0;
  qword_4FFCEB0 = 0;
  qword_4FFCEB8 = 0;
  qword_4FFCEC0 = 0;
  qword_4FFCEE0 = 0;
  qword_4FFCEE8 = (__int64)&unk_4FFCF00;
  qword_4FFCEF0 = 1;
  dword_4FFCEF8 = 0;
  byte_4FFCEFC = 1;
  v21 = sub_C57470();
  v22 = (unsigned int)qword_4FFCED0;
  if ( (unsigned __int64)(unsigned int)qword_4FFCED0 + 1 > HIDWORD(qword_4FFCED0) )
  {
    v35 = v21;
    sub_C8D5F0((char *)&unk_4FFCED8 - 16, &unk_4FFCED8, (unsigned int)qword_4FFCED0 + 1LL, 8);
    v22 = (unsigned int)qword_4FFCED0;
    v21 = v35;
  }
  *(_QWORD *)(qword_4FFCEC8 + 8 * v22) = v21;
  qword_4FFCF10 = (__int64)&unk_49D9748;
  qword_4FFCE80 = (__int64)&unk_49DC090;
  qword_4FFCF20 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FFCED0) = qword_4FFCED0 + 1;
  qword_4FFCF40 = (__int64)nullsub_23;
  qword_4FFCF08 = 0;
  qword_4FFCF38 = (__int64)sub_984030;
  qword_4FFCF18 = 0;
  sub_C53080(&qword_4FFCE80, "irce-allow-narrow-latch", 23);
  LOWORD(qword_4FFCF18) = 257;
  LOBYTE(qword_4FFCF08) = 1;
  qword_4FFCEB0 = 90;
  LOBYTE(dword_4FFCE8C) = dword_4FFCE8C & 0x9F | 0x20;
  qword_4FFCEA8 = (__int64)"If set to true, IRCE may eliminate wide range checks in loops with narrow latch condition.";
  sub_C53130(&qword_4FFCE80);
  __cxa_atexit(sub_984900, &qword_4FFCE80, &qword_4A427C0);
  qword_4FFCDA0 = (__int64)&unk_49DC150;
  v23 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FFCE1C = 1;
  word_4FFCDB0 = 0;
  qword_4FFCDF0 = 0x100000000LL;
  dword_4FFCDAC &= 0x8000u;
  qword_4FFCDE8 = (__int64)&unk_4FFCDF8;
  qword_4FFCDB8 = 0;
  dword_4FFCDA8 = v23;
  qword_4FFCDC0 = 0;
  qword_4FFCDC8 = 0;
  qword_4FFCDD0 = 0;
  qword_4FFCDD8 = 0;
  qword_4FFCDE0 = 0;
  qword_4FFCE00 = 0;
  qword_4FFCE08 = (__int64)&unk_4FFCE20;
  qword_4FFCE10 = 1;
  dword_4FFCE18 = 0;
  v24 = sub_C57470();
  v25 = (unsigned int)qword_4FFCDF0;
  if ( (unsigned __int64)(unsigned int)qword_4FFCDF0 + 1 > HIDWORD(qword_4FFCDF0) )
  {
    v36 = v24;
    sub_C8D5F0((char *)&unk_4FFCDF8 - 16, &unk_4FFCDF8, (unsigned int)qword_4FFCDF0 + 1LL, 8);
    v25 = (unsigned int)qword_4FFCDF0;
    v24 = v36;
  }
  *(_QWORD *)(qword_4FFCDE8 + 8 * v25) = v24;
  LODWORD(qword_4FFCDF0) = qword_4FFCDF0 + 1;
  qword_4FFCE28 = 0;
  qword_4FFCE30 = (__int64)&unk_49D9728;
  qword_4FFCE38 = 0;
  qword_4FFCDA0 = (__int64)&unk_49DBF10;
  qword_4FFCE40 = (__int64)&unk_49DC290;
  qword_4FFCE60 = (__int64)nullsub_24;
  qword_4FFCE58 = (__int64)sub_984050;
  sub_C53080(&qword_4FFCDA0, "irce-max-type-size-for-overflow-check", 37);
  LODWORD(qword_4FFCE28) = 32;
  BYTE4(qword_4FFCE38) = 1;
  LODWORD(qword_4FFCE38) = 32;
  qword_4FFCDD0 = 108;
  LOBYTE(dword_4FFCDAC) = dword_4FFCDAC & 0x9F | 0x20;
  qword_4FFCDC8 = (__int64)"Maximum size of range check type for which can be produced runtime overflow check of its limit's computation";
  sub_C53130(&qword_4FFCDA0);
  __cxa_atexit(sub_984970, &qword_4FFCDA0, &qword_4A427C0);
  qword_4FFCCC0 = (__int64)&unk_49DC150;
  v26 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FFCD3C = 1;
  qword_4FFCD10 = 0x100000000LL;
  dword_4FFCCCC &= 0x8000u;
  qword_4FFCD08 = (__int64)&unk_4FFCD18;
  qword_4FFCCD8 = 0;
  qword_4FFCCE0 = 0;
  dword_4FFCCC8 = v26;
  word_4FFCCD0 = 0;
  qword_4FFCCE8 = 0;
  qword_4FFCCF0 = 0;
  qword_4FFCCF8 = 0;
  qword_4FFCD00 = 0;
  qword_4FFCD20 = 0;
  qword_4FFCD28 = (__int64)&unk_4FFCD40;
  qword_4FFCD30 = 1;
  dword_4FFCD38 = 0;
  v27 = sub_C57470();
  v28 = (unsigned int)qword_4FFCD10;
  v29 = (unsigned int)qword_4FFCD10 + 1LL;
  if ( v29 > HIDWORD(qword_4FFCD10) )
  {
    sub_C8D5F0((char *)&unk_4FFCD18 - 16, &unk_4FFCD18, v29, 8);
    v28 = (unsigned int)qword_4FFCD10;
  }
  *(_QWORD *)(qword_4FFCD08 + 8 * v28) = v27;
  qword_4FFCD50 = (__int64)&unk_49D9748;
  qword_4FFCCC0 = (__int64)&unk_49DC090;
  qword_4FFCD60 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FFCD10) = qword_4FFCD10 + 1;
  qword_4FFCD80 = (__int64)nullsub_23;
  qword_4FFCD48 = 0;
  qword_4FFCD78 = (__int64)sub_984030;
  qword_4FFCD58 = 0;
  sub_C53080(&qword_4FFCCC0, "irce-print-scaled-boundary-range-checks", 39);
  LOBYTE(qword_4FFCD48) = 0;
  LOBYTE(dword_4FFCCCC) = dword_4FFCCCC & 0x9F | 0x20;
  LOWORD(qword_4FFCD58) = 256;
  sub_C53130(&qword_4FFCCC0);
  return __cxa_atexit(sub_984900, &qword_4FFCCC0, &qword_4A427C0);
}

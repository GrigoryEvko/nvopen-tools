// Function: ctor_082
// Address: 0x49eb80
//
int ctor_082()
{
  int v0; // edx
  __int64 v1; // r13
  __int64 v2; // rax
  int v3; // edx
  __int64 v4; // r13
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // edx
  __int64 v10; // r13
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // edx
  __int64 v16; // rbx
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v20; // [rsp+8h] [rbp-38h]
  __int64 v21; // [rsp+8h] [rbp-38h]

  qword_4F8F740 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8F74C = word_4F8F74C & 0x8000;
  qword_4F8F788[1] = 0x100000000LL;
  unk_4F8F748 = v0;
  unk_4F8F750 = 0;
  unk_4F8F758 = 0;
  unk_4F8F760 = 0;
  unk_4F8F768 = 0;
  unk_4F8F770 = 0;
  unk_4F8F778 = 0;
  unk_4F8F780 = 0;
  qword_4F8F788[0] = &qword_4F8F788[2];
  qword_4F8F788[3] = 0;
  qword_4F8F788[4] = &qword_4F8F788[7];
  qword_4F8F788[5] = 1;
  LODWORD(qword_4F8F788[6]) = 0;
  BYTE4(qword_4F8F788[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_4F8F788[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8F788[1]) + 1 > HIDWORD(qword_4F8F788[1]) )
  {
    sub_C8D5F0(qword_4F8F788, &qword_4F8F788[2], LODWORD(qword_4F8F788[1]) + 1LL, 8);
    v2 = LODWORD(qword_4F8F788[1]);
  }
  *(_QWORD *)(qword_4F8F788[0] + 8 * v2) = v1;
  ++LODWORD(qword_4F8F788[1]);
  qword_4F8F788[8] = 0;
  qword_4F8F788[9] = &unk_49E5940;
  qword_4F8F788[10] = 0;
  qword_4F8F740 = &unk_49E5960;
  qword_4F8F788[11] = &unk_49DC320;
  qword_4F8F788[15] = nullsub_385;
  qword_4F8F788[14] = sub_1038930;
  sub_C53080(&qword_4F8F740, "memprof-lifetime-access-density-cold-threshold", 46);
  BYTE4(qword_4F8F788[10]) = 1;
  unk_4F8F770 = 123;
  LODWORD(qword_4F8F788[8]) = 1028443341;
  LOBYTE(word_4F8F74C) = word_4F8F74C & 0x9F | 0x20;
  unk_4F8F768 = "The threshold the lifetime access density (accesses per byte per lifetime sec) must be under to consider"
                " an allocation cold";
  LODWORD(qword_4F8F788[10]) = 1028443341;
  sub_C53130(&qword_4F8F740);
  __cxa_atexit(sub_1038DB0, &qword_4F8F740, &qword_4A427C0);
  qword_4F8F660 = &unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8F66C = word_4F8F66C & 0x8000;
  qword_4F8F6A8[1] = 0x100000000LL;
  unk_4F8F668 = v3;
  unk_4F8F670 = 0;
  unk_4F8F678 = 0;
  unk_4F8F680 = 0;
  unk_4F8F688 = 0;
  unk_4F8F690 = 0;
  unk_4F8F698 = 0;
  unk_4F8F6A0 = 0;
  qword_4F8F6A8[0] = &qword_4F8F6A8[2];
  qword_4F8F6A8[3] = 0;
  qword_4F8F6A8[4] = &qword_4F8F6A8[7];
  qword_4F8F6A8[5] = 1;
  LODWORD(qword_4F8F6A8[6]) = 0;
  BYTE4(qword_4F8F6A8[6]) = 1;
  v4 = sub_C57470();
  v5 = LODWORD(qword_4F8F6A8[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8F6A8[1]) + 1 > HIDWORD(qword_4F8F6A8[1]) )
  {
    sub_C8D5F0(qword_4F8F6A8, &qword_4F8F6A8[2], LODWORD(qword_4F8F6A8[1]) + 1LL, 8);
    v5 = LODWORD(qword_4F8F6A8[1]);
  }
  *(_QWORD *)(qword_4F8F6A8[0] + 8 * v5) = v4;
  qword_4F8F6A8[9] = &unk_49D9728;
  ++LODWORD(qword_4F8F6A8[1]);
  qword_4F8F6A8[8] = 0;
  qword_4F8F660 = &unk_49DBF10;
  qword_4F8F6A8[10] = 0;
  qword_4F8F6A8[11] = &unk_49DC290;
  qword_4F8F6A8[15] = nullsub_24;
  qword_4F8F6A8[14] = sub_984050;
  sub_C53080(&qword_4F8F660, "memprof-ave-lifetime-cold-threshold", 35);
  LODWORD(qword_4F8F6A8[8]) = 200;
  BYTE4(qword_4F8F6A8[10]) = 1;
  LODWORD(qword_4F8F6A8[10]) = 200;
  unk_4F8F690 = 64;
  LOBYTE(word_4F8F66C) = word_4F8F66C & 0x9F | 0x20;
  unk_4F8F688 = "The average lifetime (s) for an allocation to be considered cold";
  sub_C53130(&qword_4F8F660);
  __cxa_atexit(sub_984970, &qword_4F8F660, &qword_4A427C0);
  qword_4F8F580 = &unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8F58C = word_4F8F58C & 0x8000;
  qword_4F8F5C8[1] = 0x100000000LL;
  unk_4F8F588 = v6;
  unk_4F8F590 = 0;
  unk_4F8F598 = 0;
  unk_4F8F5A0 = 0;
  unk_4F8F5A8 = 0;
  unk_4F8F5B0 = 0;
  unk_4F8F5B8 = 0;
  unk_4F8F5C0 = 0;
  qword_4F8F5C8[0] = &qword_4F8F5C8[2];
  qword_4F8F5C8[3] = 0;
  qword_4F8F5C8[4] = &qword_4F8F5C8[7];
  qword_4F8F5C8[5] = 1;
  LODWORD(qword_4F8F5C8[6]) = 0;
  BYTE4(qword_4F8F5C8[6]) = 1;
  v7 = sub_C57470();
  v8 = LODWORD(qword_4F8F5C8[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8F5C8[1]) + 1 > HIDWORD(qword_4F8F5C8[1]) )
  {
    v20 = v7;
    sub_C8D5F0(qword_4F8F5C8, &qword_4F8F5C8[2], LODWORD(qword_4F8F5C8[1]) + 1LL, 8);
    v8 = LODWORD(qword_4F8F5C8[1]);
    v7 = v20;
  }
  *(_QWORD *)(qword_4F8F5C8[0] + 8 * v8) = v7;
  qword_4F8F5C8[9] = &unk_49D9728;
  ++LODWORD(qword_4F8F5C8[1]);
  qword_4F8F5C8[8] = 0;
  qword_4F8F580 = &unk_49DBF10;
  qword_4F8F5C8[10] = 0;
  qword_4F8F5C8[11] = &unk_49DC290;
  qword_4F8F5C8[15] = nullsub_24;
  qword_4F8F5C8[14] = sub_984050;
  sub_C53080(&qword_4F8F580, "memprof-min-ave-lifetime-access-density-hot-threshold", 53);
  BYTE4(qword_4F8F5C8[10]) = 1;
  LODWORD(qword_4F8F5C8[8]) = 1000;
  unk_4F8F5B0 = 90;
  LODWORD(qword_4F8F5C8[10]) = 1000;
  LOBYTE(word_4F8F58C) = word_4F8F58C & 0x9F | 0x20;
  unk_4F8F5A8 = "The minimum TotalLifetimeAccessDensity / AllocCount for an allocation to be considered hot";
  sub_C53130(&qword_4F8F580);
  __cxa_atexit(sub_984970, &qword_4F8F580, &qword_4A427C0);
  qword_4F8F4A0 = &unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8F4AC = word_4F8F4AC & 0x8000;
  qword_4F8F4E8[1] = 0x100000000LL;
  unk_4F8F4A8 = v9;
  unk_4F8F4B0 = 0;
  unk_4F8F4B8 = 0;
  unk_4F8F4C0 = 0;
  unk_4F8F4C8 = 0;
  unk_4F8F4D0 = 0;
  unk_4F8F4D8 = 0;
  unk_4F8F4E0 = 0;
  qword_4F8F4E8[0] = &qword_4F8F4E8[2];
  qword_4F8F4E8[3] = 0;
  qword_4F8F4E8[4] = &qword_4F8F4E8[7];
  qword_4F8F4E8[5] = 1;
  LODWORD(qword_4F8F4E8[6]) = 0;
  BYTE4(qword_4F8F4E8[6]) = 1;
  v10 = sub_C57470();
  v11 = LODWORD(qword_4F8F4E8[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8F4E8[1]) + 1 > HIDWORD(qword_4F8F4E8[1]) )
  {
    sub_C8D5F0(qword_4F8F4E8, &qword_4F8F4E8[2], LODWORD(qword_4F8F4E8[1]) + 1LL, 8);
    v11 = LODWORD(qword_4F8F4E8[1]);
  }
  *(_QWORD *)(qword_4F8F4E8[0] + 8 * v11) = v10;
  qword_4F8F4E8[9] = &unk_49D9748;
  qword_4F8F4A0 = &unk_49DC090;
  ++LODWORD(qword_4F8F4E8[1]);
  qword_4F8F4E8[8] = 0;
  qword_4F8F4E8[11] = &unk_49DC1D0;
  qword_4F8F4E8[10] = 0;
  qword_4F8F4E8[15] = nullsub_23;
  qword_4F8F4E8[14] = sub_984030;
  sub_C53080(&qword_4F8F4A0, "memprof-use-hot-hints", 21);
  LOWORD(qword_4F8F4E8[10]) = 256;
  LOBYTE(qword_4F8F4E8[8]) = 0;
  unk_4F8F4D0 = 73;
  LOBYTE(word_4F8F4AC) = word_4F8F4AC & 0x9F | 0x20;
  unk_4F8F4C8 = "Enable use of hot hints (only supported for unambigously hot allocations)";
  sub_C53130(&qword_4F8F4A0);
  __cxa_atexit(sub_984900, &qword_4F8F4A0, &qword_4A427C0);
  qword_4F8F3C0 = &unk_49DC150;
  v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8F3CC = word_4F8F3CC & 0x8000;
  qword_4F8F408[1] = 0x100000000LL;
  unk_4F8F3C8 = v12;
  qword_4F8F408[0] = &qword_4F8F408[2];
  unk_4F8F3D0 = 0;
  unk_4F8F3D8 = 0;
  unk_4F8F3E0 = 0;
  unk_4F8F3E8 = 0;
  unk_4F8F3F0 = 0;
  unk_4F8F3F8 = 0;
  unk_4F8F400 = 0;
  qword_4F8F408[3] = 0;
  qword_4F8F408[4] = &qword_4F8F408[7];
  qword_4F8F408[5] = 1;
  LODWORD(qword_4F8F408[6]) = 0;
  BYTE4(qword_4F8F408[6]) = 1;
  v13 = sub_C57470();
  v14 = LODWORD(qword_4F8F408[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8F408[1]) + 1 > HIDWORD(qword_4F8F408[1]) )
  {
    v21 = v13;
    sub_C8D5F0(qword_4F8F408, &qword_4F8F408[2], LODWORD(qword_4F8F408[1]) + 1LL, 8);
    v14 = LODWORD(qword_4F8F408[1]);
    v13 = v21;
  }
  *(_QWORD *)(qword_4F8F408[0] + 8 * v14) = v13;
  qword_4F8F408[9] = &unk_49D9748;
  qword_4F8F3C0 = &unk_49DC090;
  ++LODWORD(qword_4F8F408[1]);
  qword_4F8F408[8] = 0;
  qword_4F8F408[11] = &unk_49DC1D0;
  qword_4F8F408[10] = 0;
  qword_4F8F408[15] = nullsub_23;
  qword_4F8F408[14] = sub_984030;
  sub_C53080(&qword_4F8F3C0, "memprof-report-hinted-sizes", 27);
  LOWORD(qword_4F8F408[10]) = 256;
  LOBYTE(qword_4F8F408[8]) = 0;
  unk_4F8F3F0 = 51;
  LOBYTE(word_4F8F3CC) = word_4F8F3CC & 0x9F | 0x20;
  unk_4F8F3E8 = "Report total allocation sizes of hinted allocations";
  sub_C53130(&qword_4F8F3C0);
  __cxa_atexit(sub_984900, &qword_4F8F3C0, &qword_4A427C0);
  qword_4F8F2E0 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8F330 = 0x100000000LL;
  dword_4F8F2EC &= 0x8000u;
  word_4F8F2F0 = 0;
  qword_4F8F328 = (__int64)&unk_4F8F338;
  qword_4F8F2F8 = 0;
  dword_4F8F2E8 = v15;
  qword_4F8F300 = 0;
  qword_4F8F308 = 0;
  qword_4F8F310 = 0;
  qword_4F8F318 = 0;
  qword_4F8F320 = 0;
  qword_4F8F340 = 0;
  qword_4F8F348 = (__int64)&unk_4F8F360;
  qword_4F8F350 = 1;
  dword_4F8F358 = 0;
  byte_4F8F35C = 1;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_4F8F330;
  v18 = (unsigned int)qword_4F8F330 + 1LL;
  if ( v18 > HIDWORD(qword_4F8F330) )
  {
    sub_C8D5F0((char *)&unk_4F8F338 - 16, &unk_4F8F338, v18, 8);
    v17 = (unsigned int)qword_4F8F330;
  }
  *(_QWORD *)(qword_4F8F328 + 8 * v17) = v16;
  qword_4F8F370 = (__int64)&unk_49D9748;
  qword_4F8F2E0 = (__int64)&unk_49DC090;
  LODWORD(qword_4F8F330) = qword_4F8F330 + 1;
  qword_4F8F368 = 0;
  qword_4F8F380 = (__int64)&unk_49DC1D0;
  qword_4F8F378 = 0;
  qword_4F8F3A0 = (__int64)nullsub_23;
  qword_4F8F398 = (__int64)sub_984030;
  sub_C53080(&qword_4F8F2E0, "memprof-keep-all-not-cold-contexts", 34);
  LOBYTE(qword_4F8F368) = 0;
  LOWORD(qword_4F8F378) = 256;
  qword_4F8F310 = 56;
  LOBYTE(dword_4F8F2EC) = dword_4F8F2EC & 0x9F | 0x20;
  qword_4F8F308 = (__int64)"Keep all non-cold contexts (increases cloning overheads)";
  sub_C53130(&qword_4F8F2E0);
  return __cxa_atexit(sub_984900, &qword_4F8F2E0, &qword_4A427C0);
}

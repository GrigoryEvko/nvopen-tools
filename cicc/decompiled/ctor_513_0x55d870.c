// Function: ctor_513
// Address: 0x55d870
//
int ctor_513()
{
  int v0; // edx
  __int64 v1; // r13
  __int64 v2; // rax
  int v3; // edx
  __int64 v4; // rax
  __int64 v5; // rdx
  int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // edx
  __int64 v13; // rbx
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v17; // [rsp+8h] [rbp-38h]
  __int64 v18; // [rsp+8h] [rbp-38h]
  __int64 v19; // [rsp+8h] [rbp-38h]

  qword_500C580 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_500C58C = word_500C58C & 0x8000;
  qword_500C5C8[1] = 0x100000000LL;
  unk_500C588 = v0;
  unk_500C590 = 0;
  unk_500C598 = 0;
  unk_500C5A0 = 0;
  unk_500C5A8 = 0;
  unk_500C5B0 = 0;
  unk_500C5B8 = 0;
  unk_500C5C0 = 0;
  qword_500C5C8[0] = &qword_500C5C8[2];
  qword_500C5C8[3] = 0;
  qword_500C5C8[4] = &qword_500C5C8[7];
  qword_500C5C8[5] = 1;
  LODWORD(qword_500C5C8[6]) = 0;
  BYTE4(qword_500C5C8[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_500C5C8[1]);
  if ( (unsigned __int64)LODWORD(qword_500C5C8[1]) + 1 > HIDWORD(qword_500C5C8[1]) )
  {
    sub_C8D5F0(qword_500C5C8, &qword_500C5C8[2], LODWORD(qword_500C5C8[1]) + 1LL, 8);
    v2 = LODWORD(qword_500C5C8[1]);
  }
  *(_QWORD *)(qword_500C5C8[0] + 8 * v2) = v1;
  qword_500C5C8[9] = &unk_49D9748;
  qword_500C580 = &unk_49DC090;
  ++LODWORD(qword_500C5C8[1]);
  qword_500C5C8[8] = 0;
  qword_500C5C8[11] = &unk_49DC1D0;
  qword_500C5C8[10] = 0;
  qword_500C5C8[15] = nullsub_23;
  qword_500C5C8[14] = sub_984030;
  sub_C53080(&qword_500C580, "vect-extend-loads", 17);
  LOWORD(qword_500C5C8[10]) = 257;
  unk_500C5B0 = 68;
  LOBYTE(qword_500C5C8[8]) = 1;
  LOBYTE(word_500C58C) = word_500C58C & 0x9F | 0x20;
  unk_500C5A8 = "Load more elements if the target VF is higher than the chain length.";
  sub_C53130(&qword_500C580);
  __cxa_atexit(sub_984900, &qword_500C580, &qword_4A427C0);
  qword_500C4A0 = &unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_500C4AC = word_500C4AC & 0x8000;
  qword_500C4E8[1] = 0x100000000LL;
  unk_500C4A8 = v3;
  qword_500C4E8[0] = &qword_500C4E8[2];
  unk_500C4B0 = 0;
  unk_500C4B8 = 0;
  unk_500C4C0 = 0;
  unk_500C4C8 = 0;
  unk_500C4D0 = 0;
  unk_500C4D8 = 0;
  unk_500C4E0 = 0;
  qword_500C4E8[3] = 0;
  qword_500C4E8[4] = &qword_500C4E8[7];
  qword_500C4E8[5] = 1;
  LODWORD(qword_500C4E8[6]) = 0;
  BYTE4(qword_500C4E8[6]) = 1;
  v4 = sub_C57470();
  v5 = LODWORD(qword_500C4E8[1]);
  if ( (unsigned __int64)LODWORD(qword_500C4E8[1]) + 1 > HIDWORD(qword_500C4E8[1]) )
  {
    v17 = v4;
    sub_C8D5F0(qword_500C4E8, &qword_500C4E8[2], LODWORD(qword_500C4E8[1]) + 1LL, 8);
    v5 = LODWORD(qword_500C4E8[1]);
    v4 = v17;
  }
  *(_QWORD *)(qword_500C4E8[0] + 8 * v5) = v4;
  qword_500C4E8[9] = &unk_49D9748;
  qword_500C4A0 = &unk_49DC090;
  ++LODWORD(qword_500C4E8[1]);
  qword_500C4E8[8] = 0;
  qword_500C4E8[11] = &unk_49DC1D0;
  qword_500C4E8[10] = 0;
  qword_500C4E8[15] = nullsub_23;
  qword_500C4E8[14] = sub_984030;
  sub_C53080(&qword_500C4A0, "vect-fill-gaps", 14);
  unk_500C4D0 = 59;
  LOWORD(qword_500C4E8[10]) = 257;
  LOBYTE(qword_500C4E8[8]) = 1;
  LOBYTE(word_500C4AC) = word_500C4AC & 0x9F | 0x20;
  unk_500C4C8 = "Should Loads be introduced in gaps to enable vectorization.";
  sub_C53130(&qword_500C4A0);
  __cxa_atexit(sub_984900, &qword_500C4A0, &qword_4A427C0);
  qword_500C3C0 = &unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_500C3CC = word_500C3CC & 0x8000;
  qword_500C408[1] = 0x100000000LL;
  unk_500C3C8 = v6;
  qword_500C408[0] = &qword_500C408[2];
  unk_500C3D0 = 0;
  unk_500C3D8 = 0;
  unk_500C3E0 = 0;
  unk_500C3E8 = 0;
  unk_500C3F0 = 0;
  unk_500C3F8 = 0;
  unk_500C400 = 0;
  qword_500C408[3] = 0;
  qword_500C408[4] = &qword_500C408[7];
  qword_500C408[5] = 1;
  LODWORD(qword_500C408[6]) = 0;
  BYTE4(qword_500C408[6]) = 1;
  v7 = sub_C57470();
  v8 = LODWORD(qword_500C408[1]);
  if ( (unsigned __int64)LODWORD(qword_500C408[1]) + 1 > HIDWORD(qword_500C408[1]) )
  {
    v18 = v7;
    sub_C8D5F0(qword_500C408, &qword_500C408[2], LODWORD(qword_500C408[1]) + 1LL, 8);
    v8 = LODWORD(qword_500C408[1]);
    v7 = v18;
  }
  *(_QWORD *)(qword_500C408[0] + 8 * v8) = v7;
  qword_500C408[9] = &unk_49D9748;
  qword_500C3C0 = &unk_49DC090;
  ++LODWORD(qword_500C408[1]);
  qword_500C408[8] = 0;
  qword_500C408[11] = &unk_49DC1D0;
  qword_500C408[10] = 0;
  qword_500C408[15] = nullsub_23;
  qword_500C408[14] = sub_984030;
  sub_C53080(&qword_500C3C0, "vect-align-scev", 15);
  unk_500C3E8 = "Refine alignment of load/store pointers using SCEV";
  LOWORD(qword_500C408[10]) = 257;
  unk_500C3F0 = 50;
  LOBYTE(qword_500C408[8]) = 1;
  LOBYTE(word_500C3CC) = word_500C3CC & 0x9F | 0x20;
  sub_C53130(&qword_500C3C0);
  __cxa_atexit(sub_984900, &qword_500C3C0, &qword_4A427C0);
  qword_500C2E0 = (__int64)&unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_500C35C = 1;
  word_500C2F0 = 0;
  qword_500C330 = 0x100000000LL;
  dword_500C2EC &= 0x8000u;
  qword_500C328 = (__int64)&unk_500C338;
  qword_500C2F8 = 0;
  dword_500C2E8 = v9;
  qword_500C300 = 0;
  qword_500C308 = 0;
  qword_500C310 = 0;
  qword_500C318 = 0;
  qword_500C320 = 0;
  qword_500C340 = 0;
  qword_500C348 = (__int64)&unk_500C360;
  qword_500C350 = 1;
  dword_500C358 = 0;
  v10 = sub_C57470();
  v11 = (unsigned int)qword_500C330;
  if ( (unsigned __int64)(unsigned int)qword_500C330 + 1 > HIDWORD(qword_500C330) )
  {
    v19 = v10;
    sub_C8D5F0((char *)&unk_500C338 - 16, &unk_500C338, (unsigned int)qword_500C330 + 1LL, 8);
    v11 = (unsigned int)qword_500C330;
    v10 = v19;
  }
  *(_QWORD *)(qword_500C328 + 8 * v11) = v10;
  qword_500C370 = (__int64)&unk_49D9748;
  qword_500C2E0 = (__int64)&unk_49DC090;
  LODWORD(qword_500C330) = qword_500C330 + 1;
  qword_500C368 = 0;
  qword_500C380 = (__int64)&unk_49DC1D0;
  qword_500C378 = 0;
  qword_500C3A0 = (__int64)nullsub_23;
  qword_500C398 = (__int64)sub_984030;
  sub_C53080(&qword_500C2E0, "vect-intrinsics", 15);
  qword_500C310 = 77;
  LOBYTE(qword_500C368) = 1;
  LOBYTE(dword_500C2EC) = dword_500C2EC & 0x9F | 0x20;
  qword_500C308 = (__int64)"Perform vectorization of nvvm_load/nvvm_ld and nvvm_store/nvvm_st intrinsics.";
  LOWORD(qword_500C378) = 257;
  sub_C53130(&qword_500C2E0);
  __cxa_atexit(sub_984900, &qword_500C2E0, &qword_4A427C0);
  qword_500C200 = (__int64)&unk_49DC150;
  v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_500C27C = 1;
  qword_500C250 = 0x100000000LL;
  dword_500C20C &= 0x8000u;
  qword_500C218 = 0;
  qword_500C220 = 0;
  qword_500C228 = 0;
  dword_500C208 = v12;
  word_500C210 = 0;
  qword_500C230 = 0;
  qword_500C238 = 0;
  qword_500C240 = 0;
  qword_500C248 = (__int64)&unk_500C258;
  qword_500C260 = 0;
  qword_500C268 = (__int64)&unk_500C280;
  qword_500C270 = 1;
  dword_500C278 = 0;
  v13 = sub_C57470();
  v14 = (unsigned int)qword_500C250;
  v15 = (unsigned int)qword_500C250 + 1LL;
  if ( v15 > HIDWORD(qword_500C250) )
  {
    sub_C8D5F0((char *)&unk_500C258 - 16, &unk_500C258, v15, 8);
    v14 = (unsigned int)qword_500C250;
  }
  *(_QWORD *)(qword_500C248 + 8 * v14) = v13;
  LODWORD(qword_500C250) = qword_500C250 + 1;
  qword_500C288 = 0;
  qword_500C290 = (__int64)&unk_49D9728;
  qword_500C298 = 0;
  qword_500C200 = (__int64)&unk_49DBF10;
  qword_500C2A0 = (__int64)&unk_49DC290;
  qword_500C2C0 = (__int64)nullsub_24;
  qword_500C2B8 = (__int64)sub_984050;
  sub_C53080(&qword_500C200, "vect-get-underlying-maxlookup", 29);
  LODWORD(qword_500C288) = 12;
  BYTE4(qword_500C298) = 1;
  LODWORD(qword_500C298) = 12;
  qword_500C228 = (__int64)"Depth that getUnderlyingObject searches";
  qword_500C230 = 39;
  sub_C53130(&qword_500C200);
  return __cxa_atexit(sub_984970, &qword_500C200, &qword_4A427C0);
}

// Function: ctor_680
// Address: 0x5a4940
//
int __fastcall ctor_680(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r15
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rcx
  int v21; // edx
  __int64 v22; // r15
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rcx
  int v27; // edx
  __int64 v28; // rbx
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __int64 v32; // [rsp+8h] [rbp-38h]

  qword_503F5E0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_503F630 = 0x100000000LL;
  dword_503F5EC &= 0x8000u;
  word_503F5F0 = 0;
  qword_503F5F8 = 0;
  qword_503F600 = 0;
  dword_503F5E8 = v4;
  qword_503F608 = 0;
  qword_503F610 = 0;
  qword_503F618 = 0;
  qword_503F620 = 0;
  qword_503F628 = (__int64)&unk_503F638;
  qword_503F640 = 0;
  qword_503F648 = (__int64)&unk_503F660;
  qword_503F650 = 1;
  dword_503F658 = 0;
  byte_503F65C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_503F630;
  v7 = (unsigned int)qword_503F630 + 1LL;
  if ( v7 > HIDWORD(qword_503F630) )
  {
    sub_C8D5F0((char *)&unk_503F638 - 16, &unk_503F638, v7, 8);
    v6 = (unsigned int)qword_503F630;
  }
  *(_QWORD *)(qword_503F628 + 8 * v6) = v5;
  qword_503F670 = (__int64)&unk_49D9748;
  qword_503F5E0 = (__int64)&unk_49DC090;
  LODWORD(qword_503F630) = qword_503F630 + 1;
  qword_503F668 = 0;
  qword_503F680 = (__int64)&unk_49DC1D0;
  qword_503F678 = 0;
  qword_503F6A0 = (__int64)nullsub_23;
  qword_503F698 = (__int64)sub_984030;
  sub_C53080(&qword_503F5E0, "show-fs-branchprob", 18);
  LOWORD(qword_503F678) = 256;
  LOBYTE(qword_503F668) = 0;
  qword_503F610 = 49;
  LOBYTE(dword_503F5EC) = dword_503F5EC & 0x9F | 0x20;
  qword_503F608 = (__int64)"Print setting flow sensitive branch probabilities";
  sub_C53130(&qword_503F5E0);
  __cxa_atexit(sub_984900, &qword_503F5E0, &qword_4A427C0);
  qword_503F500 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_503F5E0, v8, v9), 1u);
  qword_503F550 = 0x100000000LL;
  dword_503F50C &= 0x8000u;
  qword_503F548 = (__int64)&unk_503F558;
  word_503F510 = 0;
  qword_503F518 = 0;
  dword_503F508 = v10;
  qword_503F520 = 0;
  qword_503F528 = 0;
  qword_503F530 = 0;
  qword_503F538 = 0;
  qword_503F540 = 0;
  qword_503F560 = 0;
  qword_503F568 = (__int64)&unk_503F580;
  qword_503F570 = 1;
  dword_503F578 = 0;
  byte_503F57C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_503F550;
  v13 = (unsigned int)qword_503F550 + 1LL;
  if ( v13 > HIDWORD(qword_503F550) )
  {
    sub_C8D5F0((char *)&unk_503F558 - 16, &unk_503F558, v13, 8);
    v12 = (unsigned int)qword_503F550;
  }
  *(_QWORD *)(qword_503F548 + 8 * v12) = v11;
  LODWORD(qword_503F550) = qword_503F550 + 1;
  qword_503F588 = 0;
  qword_503F590 = (__int64)&unk_49D9728;
  qword_503F598 = 0;
  qword_503F500 = (__int64)&unk_49DBF10;
  qword_503F5A0 = (__int64)&unk_49DC290;
  qword_503F5C0 = (__int64)nullsub_24;
  qword_503F5B8 = (__int64)sub_984050;
  sub_C53080(&qword_503F500, "fs-profile-debug-prob-diff-threshold", 36);
  LODWORD(qword_503F588) = 10;
  qword_503F528 = (__int64)"Only show debug message if the branch probability is greater than this value (in percentage).";
  BYTE4(qword_503F598) = 1;
  LODWORD(qword_503F598) = 10;
  qword_503F530 = 93;
  sub_C53130(&qword_503F500);
  __cxa_atexit(sub_984970, &qword_503F500, &qword_4A427C0);
  qword_503F420 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_503F500, v14, v15), 1u);
  qword_503F470 = 0x100000000LL;
  dword_503F42C &= 0x8000u;
  qword_503F468 = (__int64)&unk_503F478;
  word_503F430 = 0;
  qword_503F438 = 0;
  dword_503F428 = v16;
  qword_503F440 = 0;
  qword_503F448 = 0;
  qword_503F450 = 0;
  qword_503F458 = 0;
  qword_503F460 = 0;
  qword_503F480 = 0;
  qword_503F488 = (__int64)&unk_503F4A0;
  qword_503F490 = 1;
  dword_503F498 = 0;
  byte_503F49C = 1;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_503F470;
  if ( (unsigned __int64)(unsigned int)qword_503F470 + 1 > HIDWORD(qword_503F470) )
  {
    v32 = v17;
    sub_C8D5F0((char *)&unk_503F478 - 16, &unk_503F478, (unsigned int)qword_503F470 + 1LL, 8);
    v18 = (unsigned int)qword_503F470;
    v17 = v32;
  }
  *(_QWORD *)(qword_503F468 + 8 * v18) = v17;
  LODWORD(qword_503F470) = qword_503F470 + 1;
  qword_503F4A8 = 0;
  qword_503F4B0 = (__int64)&unk_49D9728;
  qword_503F4B8 = 0;
  qword_503F420 = (__int64)&unk_49DBF10;
  qword_503F4C0 = (__int64)&unk_49DC290;
  qword_503F4E0 = (__int64)nullsub_24;
  qword_503F4D8 = (__int64)sub_984050;
  sub_C53080(&qword_503F420, "fs-profile-debug-bw-threshold", 29);
  LODWORD(qword_503F4A8) = 10000;
  qword_503F448 = (__int64)"Only show debug message if the source branch weight is greater  than this value.";
  BYTE4(qword_503F4B8) = 1;
  LODWORD(qword_503F4B8) = 10000;
  qword_503F450 = 80;
  sub_C53130(&qword_503F420);
  __cxa_atexit(sub_984970, &qword_503F420, &qword_4A427C0);
  qword_503F340 = (__int64)&unk_49DC150;
  v21 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_503F420, v19, v20), 1u);
  byte_503F3BC = 1;
  word_503F350 = 0;
  qword_503F390 = 0x100000000LL;
  dword_503F34C &= 0x8000u;
  qword_503F388 = (__int64)&unk_503F398;
  qword_503F358 = 0;
  dword_503F348 = v21;
  qword_503F360 = 0;
  qword_503F368 = 0;
  qword_503F370 = 0;
  qword_503F378 = 0;
  qword_503F380 = 0;
  qword_503F3A0 = 0;
  qword_503F3A8 = (__int64)&unk_503F3C0;
  qword_503F3B0 = 1;
  dword_503F3B8 = 0;
  v22 = sub_C57470();
  v23 = (unsigned int)qword_503F390;
  v24 = (unsigned int)qword_503F390 + 1LL;
  if ( v24 > HIDWORD(qword_503F390) )
  {
    sub_C8D5F0((char *)&unk_503F398 - 16, &unk_503F398, v24, 8);
    v23 = (unsigned int)qword_503F390;
  }
  *(_QWORD *)(qword_503F388 + 8 * v23) = v22;
  qword_503F3D0 = (__int64)&unk_49D9748;
  qword_503F340 = (__int64)&unk_49DC090;
  LODWORD(qword_503F390) = qword_503F390 + 1;
  qword_503F3C8 = 0;
  qword_503F3E0 = (__int64)&unk_49DC1D0;
  qword_503F3D8 = 0;
  qword_503F400 = (__int64)nullsub_23;
  qword_503F3F8 = (__int64)sub_984030;
  sub_C53080(&qword_503F340, "fs-viewbfi-before", 17);
  LOWORD(qword_503F3D8) = 256;
  LOBYTE(qword_503F3C8) = 0;
  qword_503F370 = 26;
  LOBYTE(dword_503F34C) = dword_503F34C & 0x9F | 0x20;
  qword_503F368 = (__int64)"View BFI before MIR loader";
  sub_C53130(&qword_503F340);
  __cxa_atexit(sub_984900, &qword_503F340, &qword_4A427C0);
  qword_503F260 = (__int64)&unk_49DC150;
  v27 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_503F340, v25, v26), 1u);
  qword_503F2B0 = 0x100000000LL;
  word_503F270 = 0;
  dword_503F26C &= 0x8000u;
  qword_503F278 = 0;
  qword_503F280 = 0;
  dword_503F268 = v27;
  qword_503F288 = 0;
  qword_503F290 = 0;
  qword_503F298 = 0;
  qword_503F2A0 = 0;
  qword_503F2A8 = (__int64)&unk_503F2B8;
  qword_503F2C0 = 0;
  qword_503F2C8 = (__int64)&unk_503F2E0;
  qword_503F2D0 = 1;
  dword_503F2D8 = 0;
  byte_503F2DC = 1;
  v28 = sub_C57470();
  v29 = (unsigned int)qword_503F2B0;
  v30 = (unsigned int)qword_503F2B0 + 1LL;
  if ( v30 > HIDWORD(qword_503F2B0) )
  {
    sub_C8D5F0((char *)&unk_503F2B8 - 16, &unk_503F2B8, v30, 8);
    v29 = (unsigned int)qword_503F2B0;
  }
  *(_QWORD *)(qword_503F2A8 + 8 * v29) = v28;
  qword_503F2F0 = (__int64)&unk_49D9748;
  qword_503F260 = (__int64)&unk_49DC090;
  LODWORD(qword_503F2B0) = qword_503F2B0 + 1;
  qword_503F2E8 = 0;
  qword_503F300 = (__int64)&unk_49DC1D0;
  qword_503F2F8 = 0;
  qword_503F320 = (__int64)nullsub_23;
  qword_503F318 = (__int64)sub_984030;
  sub_C53080(&qword_503F260, "fs-viewbfi-after", 16);
  LOBYTE(qword_503F2E8) = 0;
  qword_503F290 = 25;
  LOBYTE(dword_503F26C) = dword_503F26C & 0x9F | 0x20;
  LOWORD(qword_503F2F8) = 256;
  qword_503F288 = (__int64)"View BFI after MIR loader";
  sub_C53130(&qword_503F260);
  return __cxa_atexit(sub_984900, &qword_503F260, &qword_4A427C0);
}

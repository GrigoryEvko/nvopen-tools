// Function: ctor_096_0
// Address: 0x4a2e30
//
int ctor_096_0()
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
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // edx
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // edx
  __int64 v19; // rax
  __int64 v20; // rdx
  int v21; // edx
  __int64 v22; // r13
  __int64 v23; // rax
  int v24; // edx
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v28; // [rsp+8h] [rbp-38h]
  __int64 v29; // [rsp+8h] [rbp-38h]
  __int64 v30; // [rsp+8h] [rbp-38h]
  __int64 v31; // [rsp+8h] [rbp-38h]
  __int64 v32; // [rsp+8h] [rbp-38h]
  __int64 v33; // [rsp+8h] [rbp-38h]

  qword_4F91B60 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F91B6C = word_4F91B6C & 0x8000;
  qword_4F91BA8[1] = 0x100000000LL;
  unk_4F91B68 = v0;
  unk_4F91B70 = 0;
  unk_4F91B78 = 0;
  unk_4F91B80 = 0;
  unk_4F91B88 = 0;
  unk_4F91B90 = 0;
  unk_4F91B98 = 0;
  unk_4F91BA0 = 0;
  qword_4F91BA8[0] = &qword_4F91BA8[2];
  qword_4F91BA8[3] = 0;
  qword_4F91BA8[4] = &qword_4F91BA8[7];
  qword_4F91BA8[5] = 1;
  LODWORD(qword_4F91BA8[6]) = 0;
  BYTE4(qword_4F91BA8[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_4F91BA8[1]);
  if ( (unsigned __int64)LODWORD(qword_4F91BA8[1]) + 1 > HIDWORD(qword_4F91BA8[1]) )
  {
    sub_C8D5F0(qword_4F91BA8, &qword_4F91BA8[2], LODWORD(qword_4F91BA8[1]) + 1LL, 8);
    v2 = LODWORD(qword_4F91BA8[1]);
  }
  *(_QWORD *)(qword_4F91BA8[0] + 8 * v2) = v1;
  qword_4F91BA8[9] = &unk_49D9748;
  qword_4F91B60 = &unk_49DC090;
  ++LODWORD(qword_4F91BA8[1]);
  qword_4F91BA8[8] = 0;
  qword_4F91BA8[11] = &unk_49DC1D0;
  qword_4F91BA8[10] = 0;
  qword_4F91BA8[15] = nullsub_23;
  qword_4F91BA8[14] = sub_984030;
  sub_C53080(&qword_4F91B60, "pgso", 4);
  LOBYTE(qword_4F91BA8[8]) = 1;
  unk_4F91B90 = 46;
  LOBYTE(word_4F91B6C) = word_4F91B6C & 0x9F | 0x20;
  LOWORD(qword_4F91BA8[10]) = 257;
  unk_4F91B88 = "Enable the profile guided size optimizations. ";
  sub_C53130(&qword_4F91B60);
  __cxa_atexit(sub_984900, &qword_4F91B60, &qword_4A427C0);
  qword_4F91A80 = &unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F91A8C = word_4F91A8C & 0x8000;
  unk_4F91A88 = v3;
  qword_4F91AC8[1] = 0x100000000LL;
  unk_4F91A90 = 0;
  qword_4F91AC8[0] = &qword_4F91AC8[2];
  unk_4F91A98 = 0;
  unk_4F91AA0 = 0;
  unk_4F91AA8 = 0;
  unk_4F91AB0 = 0;
  unk_4F91AB8 = 0;
  unk_4F91AC0 = 0;
  qword_4F91AC8[3] = 0;
  qword_4F91AC8[4] = &qword_4F91AC8[7];
  qword_4F91AC8[5] = 1;
  LODWORD(qword_4F91AC8[6]) = 0;
  BYTE4(qword_4F91AC8[6]) = 1;
  v4 = sub_C57470();
  v5 = LODWORD(qword_4F91AC8[1]);
  if ( (unsigned __int64)LODWORD(qword_4F91AC8[1]) + 1 > HIDWORD(qword_4F91AC8[1]) )
  {
    v28 = v4;
    sub_C8D5F0(qword_4F91AC8, &qword_4F91AC8[2], LODWORD(qword_4F91AC8[1]) + 1LL, 8);
    v5 = LODWORD(qword_4F91AC8[1]);
    v4 = v28;
  }
  *(_QWORD *)(qword_4F91AC8[0] + 8 * v5) = v4;
  qword_4F91AC8[9] = &unk_49D9748;
  qword_4F91A80 = &unk_49DC090;
  ++LODWORD(qword_4F91AC8[1]);
  qword_4F91AC8[8] = 0;
  qword_4F91AC8[11] = &unk_49DC1D0;
  qword_4F91AC8[10] = 0;
  qword_4F91AC8[15] = nullsub_23;
  qword_4F91AC8[14] = sub_984030;
  sub_C53080(&qword_4F91A80, "pgso-lwss-only", 14);
  LOBYTE(qword_4F91AC8[8]) = 1;
  unk_4F91AB0 = 105;
  LOBYTE(word_4F91A8C) = word_4F91A8C & 0x9F | 0x20;
  LOWORD(qword_4F91AC8[10]) = 257;
  unk_4F91AA8 = "Apply the profile guided size optimizations only if the working set size is large (except for cold code.)";
  sub_C53130(&qword_4F91A80);
  __cxa_atexit(sub_984900, &qword_4F91A80, &qword_4A427C0);
  qword_4F919A0 = &unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F919AC = word_4F919AC & 0x8000;
  unk_4F919B0 = 0;
  qword_4F919E8[1] = 0x100000000LL;
  unk_4F919A8 = v6;
  qword_4F919E8[0] = &qword_4F919E8[2];
  unk_4F919B8 = 0;
  unk_4F919C0 = 0;
  unk_4F919C8 = 0;
  unk_4F919D0 = 0;
  unk_4F919D8 = 0;
  unk_4F919E0 = 0;
  qword_4F919E8[3] = 0;
  qword_4F919E8[4] = &qword_4F919E8[7];
  qword_4F919E8[5] = 1;
  LODWORD(qword_4F919E8[6]) = 0;
  BYTE4(qword_4F919E8[6]) = 1;
  v7 = sub_C57470();
  v8 = LODWORD(qword_4F919E8[1]);
  if ( (unsigned __int64)LODWORD(qword_4F919E8[1]) + 1 > HIDWORD(qword_4F919E8[1]) )
  {
    v29 = v7;
    sub_C8D5F0(qword_4F919E8, &qword_4F919E8[2], LODWORD(qword_4F919E8[1]) + 1LL, 8);
    v8 = LODWORD(qword_4F919E8[1]);
    v7 = v29;
  }
  *(_QWORD *)(qword_4F919E8[0] + 8 * v8) = v7;
  qword_4F919E8[9] = &unk_49D9748;
  qword_4F919A0 = &unk_49DC090;
  ++LODWORD(qword_4F919E8[1]);
  qword_4F919E8[8] = 0;
  qword_4F919E8[11] = &unk_49DC1D0;
  qword_4F919E8[10] = 0;
  qword_4F919E8[15] = nullsub_23;
  qword_4F919E8[14] = sub_984030;
  sub_C53080(&qword_4F919A0, "pgso-cold-code-only", 19);
  LOBYTE(qword_4F919E8[8]) = 0;
  unk_4F919D0 = 62;
  LOBYTE(word_4F919AC) = word_4F919AC & 0x9F | 0x20;
  LOWORD(qword_4F919E8[10]) = 256;
  unk_4F919C8 = "Apply the profile guided size optimizations only to cold code.";
  sub_C53130(&qword_4F919A0);
  __cxa_atexit(sub_984900, &qword_4F919A0, &qword_4A427C0);
  qword_4F918C0 = &unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F918CC = word_4F918CC & 0x8000;
  unk_4F918D0 = 0;
  qword_4F91908[1] = 0x100000000LL;
  unk_4F918C8 = v9;
  qword_4F91908[0] = &qword_4F91908[2];
  unk_4F918D8 = 0;
  unk_4F918E0 = 0;
  unk_4F918E8 = 0;
  unk_4F918F0 = 0;
  unk_4F918F8 = 0;
  unk_4F91900 = 0;
  qword_4F91908[3] = 0;
  qword_4F91908[4] = &qword_4F91908[7];
  qword_4F91908[5] = 1;
  LODWORD(qword_4F91908[6]) = 0;
  BYTE4(qword_4F91908[6]) = 1;
  v10 = sub_C57470();
  v11 = LODWORD(qword_4F91908[1]);
  if ( (unsigned __int64)LODWORD(qword_4F91908[1]) + 1 > HIDWORD(qword_4F91908[1]) )
  {
    v30 = v10;
    sub_C8D5F0(qword_4F91908, &qword_4F91908[2], LODWORD(qword_4F91908[1]) + 1LL, 8);
    v11 = LODWORD(qword_4F91908[1]);
    v10 = v30;
  }
  *(_QWORD *)(qword_4F91908[0] + 8 * v11) = v10;
  qword_4F91908[9] = &unk_49D9748;
  qword_4F918C0 = &unk_49DC090;
  ++LODWORD(qword_4F91908[1]);
  qword_4F91908[8] = 0;
  qword_4F91908[11] = &unk_49DC1D0;
  qword_4F91908[10] = 0;
  qword_4F91908[15] = nullsub_23;
  qword_4F91908[14] = sub_984030;
  sub_C53080(&qword_4F918C0, "pgso-cold-code-only-for-instr-pgo", 33);
  LOWORD(qword_4F91908[10]) = 256;
  LOBYTE(qword_4F91908[8]) = 0;
  unk_4F918F0 = 88;
  LOBYTE(word_4F918CC) = word_4F918CC & 0x9F | 0x20;
  unk_4F918E8 = "Apply the profile guided size optimizations only to cold code under instrumentation PGO.";
  sub_C53130(&qword_4F918C0);
  __cxa_atexit(sub_984900, &qword_4F918C0, &qword_4A427C0);
  qword_4F917E0 = &unk_49DC150;
  v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F917EC = word_4F917EC & 0x8000;
  qword_4F91828[1] = 0x100000000LL;
  unk_4F917E8 = v12;
  qword_4F91828[0] = &qword_4F91828[2];
  unk_4F917F0 = 0;
  unk_4F917F8 = 0;
  unk_4F91800 = 0;
  unk_4F91808 = 0;
  unk_4F91810 = 0;
  unk_4F91818 = 0;
  unk_4F91820 = 0;
  qword_4F91828[3] = 0;
  qword_4F91828[4] = &qword_4F91828[7];
  qword_4F91828[5] = 1;
  LODWORD(qword_4F91828[6]) = 0;
  BYTE4(qword_4F91828[6]) = 1;
  v13 = sub_C57470();
  v14 = LODWORD(qword_4F91828[1]);
  if ( (unsigned __int64)LODWORD(qword_4F91828[1]) + 1 > HIDWORD(qword_4F91828[1]) )
  {
    v31 = v13;
    sub_C8D5F0(qword_4F91828, &qword_4F91828[2], LODWORD(qword_4F91828[1]) + 1LL, 8);
    v14 = LODWORD(qword_4F91828[1]);
    v13 = v31;
  }
  *(_QWORD *)(qword_4F91828[0] + 8 * v14) = v13;
  qword_4F91828[9] = &unk_49D9748;
  qword_4F917E0 = &unk_49DC090;
  ++LODWORD(qword_4F91828[1]);
  qword_4F91828[8] = 0;
  qword_4F91828[11] = &unk_49DC1D0;
  qword_4F91828[10] = 0;
  qword_4F91828[15] = nullsub_23;
  qword_4F91828[14] = sub_984030;
  sub_C53080(&qword_4F917E0, "pgso-cold-code-only-for-sample-pgo", 34);
  LOWORD(qword_4F91828[10]) = 256;
  LOBYTE(qword_4F91828[8]) = 0;
  unk_4F91810 = 79;
  LOBYTE(word_4F917EC) = word_4F917EC & 0x9F | 0x20;
  unk_4F91808 = "Apply the profile guided size optimizations only to cold code under sample PGO.";
  sub_C53130(&qword_4F917E0);
  __cxa_atexit(sub_984900, &qword_4F917E0, &qword_4A427C0);
  qword_4F91700 = &unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F9170C = word_4F9170C & 0x8000;
  qword_4F91748[1] = 0x100000000LL;
  unk_4F91708 = v15;
  unk_4F91710 = 0;
  qword_4F91748[0] = &qword_4F91748[2];
  unk_4F91718 = 0;
  unk_4F91720 = 0;
  unk_4F91728 = 0;
  unk_4F91730 = 0;
  unk_4F91738 = 0;
  unk_4F91740 = 0;
  qword_4F91748[3] = 0;
  qword_4F91748[4] = &qword_4F91748[7];
  qword_4F91748[5] = 1;
  LODWORD(qword_4F91748[6]) = 0;
  BYTE4(qword_4F91748[6]) = 1;
  v16 = sub_C57470();
  v17 = LODWORD(qword_4F91748[1]);
  if ( (unsigned __int64)LODWORD(qword_4F91748[1]) + 1 > HIDWORD(qword_4F91748[1]) )
  {
    v32 = v16;
    sub_C8D5F0(qword_4F91748, &qword_4F91748[2], LODWORD(qword_4F91748[1]) + 1LL, 8);
    v17 = LODWORD(qword_4F91748[1]);
    v16 = v32;
  }
  *(_QWORD *)(qword_4F91748[0] + 8 * v17) = v16;
  qword_4F91748[9] = &unk_49D9748;
  qword_4F91700 = &unk_49DC090;
  ++LODWORD(qword_4F91748[1]);
  qword_4F91748[8] = 0;
  qword_4F91748[11] = &unk_49DC1D0;
  qword_4F91748[10] = 0;
  qword_4F91748[15] = nullsub_23;
  qword_4F91748[14] = sub_984030;
  sub_C53080(&qword_4F91700, "pgso-cold-code-only-for-partial-sample-pgo", 42);
  LOWORD(qword_4F91748[10]) = 256;
  LOBYTE(qword_4F91748[8]) = 0;
  unk_4F91730 = 95;
  LOBYTE(word_4F9170C) = word_4F9170C & 0x9F | 0x20;
  unk_4F91728 = "Apply the profile guided size optimizations only to cold code under partial-profile sample PGO.";
  sub_C53130(&qword_4F91700);
  __cxa_atexit(sub_984900, &qword_4F91700, &qword_4A427C0);
  qword_4F91620 = &unk_49DC150;
  v18 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F9162C = word_4F9162C & 0x8000;
  qword_4F91668[1] = 0x100000000LL;
  unk_4F91628 = v18;
  qword_4F91668[0] = &qword_4F91668[2];
  unk_4F91630 = 0;
  unk_4F91638 = 0;
  unk_4F91640 = 0;
  unk_4F91648 = 0;
  unk_4F91650 = 0;
  unk_4F91658 = 0;
  unk_4F91660 = 0;
  qword_4F91668[3] = 0;
  qword_4F91668[4] = &qword_4F91668[7];
  qword_4F91668[5] = 1;
  LODWORD(qword_4F91668[6]) = 0;
  BYTE4(qword_4F91668[6]) = 1;
  v19 = sub_C57470();
  v20 = LODWORD(qword_4F91668[1]);
  if ( (unsigned __int64)LODWORD(qword_4F91668[1]) + 1 > HIDWORD(qword_4F91668[1]) )
  {
    v33 = v19;
    sub_C8D5F0(qword_4F91668, &qword_4F91668[2], LODWORD(qword_4F91668[1]) + 1LL, 8);
    v20 = LODWORD(qword_4F91668[1]);
    v19 = v33;
  }
  *(_QWORD *)(qword_4F91668[0] + 8 * v20) = v19;
  qword_4F91668[9] = &unk_49D9748;
  qword_4F91620 = &unk_49DC090;
  ++LODWORD(qword_4F91668[1]);
  qword_4F91668[8] = 0;
  qword_4F91668[11] = &unk_49DC1D0;
  qword_4F91668[10] = 0;
  qword_4F91668[15] = nullsub_23;
  qword_4F91668[14] = sub_984030;
  sub_C53080(&qword_4F91620, "force-pgso", 10);
  LOWORD(qword_4F91668[10]) = 256;
  LOBYTE(qword_4F91668[8]) = 0;
  unk_4F91650 = 48;
  LOBYTE(word_4F9162C) = word_4F9162C & 0x9F | 0x20;
  unk_4F91648 = "Force the (profiled-guided) size optimizations. ";
  sub_C53130(&qword_4F91620);
  __cxa_atexit(sub_984900, &qword_4F91620, &qword_4A427C0);
  qword_4F91540 = &unk_49DC150;
  v21 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F9154C = word_4F9154C & 0x8000;
  qword_4F91588[1] = 0x100000000LL;
  unk_4F91548 = v21;
  unk_4F91550 = 0;
  unk_4F91558 = 0;
  unk_4F91560 = 0;
  unk_4F91568 = 0;
  unk_4F91570 = 0;
  unk_4F91578 = 0;
  unk_4F91580 = 0;
  qword_4F91588[0] = &qword_4F91588[2];
  qword_4F91588[3] = 0;
  qword_4F91588[4] = &qword_4F91588[7];
  qword_4F91588[5] = 1;
  LODWORD(qword_4F91588[6]) = 0;
  BYTE4(qword_4F91588[6]) = 1;
  v22 = sub_C57470();
  v23 = LODWORD(qword_4F91588[1]);
  if ( (unsigned __int64)LODWORD(qword_4F91588[1]) + 1 > HIDWORD(qword_4F91588[1]) )
  {
    sub_C8D5F0(qword_4F91588, &qword_4F91588[2], LODWORD(qword_4F91588[1]) + 1LL, 8);
    v23 = LODWORD(qword_4F91588[1]);
  }
  *(_QWORD *)(qword_4F91588[0] + 8 * v23) = v22;
  qword_4F91588[9] = &unk_49DA090;
  ++LODWORD(qword_4F91588[1]);
  qword_4F91588[8] = 0;
  qword_4F91540 = &unk_49DBF90;
  qword_4F91588[10] = 0;
  qword_4F91588[11] = &unk_49DC230;
  qword_4F91588[15] = nullsub_58;
  qword_4F91588[14] = sub_B2B5F0;
  sub_C53080(&qword_4F91540, "pgso-cutoff-instr-prof", 22);
  LODWORD(qword_4F91588[8]) = 950000;
  BYTE4(qword_4F91588[10]) = 1;
  LODWORD(qword_4F91588[10]) = 950000;
  unk_4F91570 = 88;
  LOBYTE(word_4F9154C) = word_4F9154C & 0x9F | 0x20;
  unk_4F91568 = "The profile guided size optimization profile summary cutoff for instrumentation profile.";
  sub_C53130(&qword_4F91540);
  __cxa_atexit(sub_B2B680, &qword_4F91540, &qword_4A427C0);
  qword_4F91460 = &unk_49DC150;
  v24 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F9146C = word_4F9146C & 0x8000;
  unk_4F91470 = 0;
  qword_4F914A8[1] = 0x100000000LL;
  unk_4F91468 = v24;
  unk_4F91478 = 0;
  unk_4F91480 = 0;
  unk_4F91488 = 0;
  unk_4F91490 = 0;
  unk_4F91498 = 0;
  unk_4F914A0 = 0;
  qword_4F914A8[0] = &qword_4F914A8[2];
  qword_4F914A8[3] = 0;
  qword_4F914A8[4] = &qword_4F914A8[7];
  qword_4F914A8[5] = 1;
  LODWORD(qword_4F914A8[6]) = 0;
  BYTE4(qword_4F914A8[6]) = 1;
  v25 = sub_C57470();
  v26 = LODWORD(qword_4F914A8[1]);
  if ( (unsigned __int64)LODWORD(qword_4F914A8[1]) + 1 > HIDWORD(qword_4F914A8[1]) )
  {
    sub_C8D5F0(qword_4F914A8, &qword_4F914A8[2], LODWORD(qword_4F914A8[1]) + 1LL, 8);
    v26 = LODWORD(qword_4F914A8[1]);
  }
  *(_QWORD *)(qword_4F914A8[0] + 8 * v26) = v25;
  qword_4F914A8[9] = &unk_49DA090;
  ++LODWORD(qword_4F914A8[1]);
  qword_4F914A8[8] = 0;
  qword_4F91460 = &unk_49DBF90;
  qword_4F914A8[10] = 0;
  qword_4F914A8[11] = &unk_49DC230;
  qword_4F914A8[15] = nullsub_58;
  qword_4F914A8[14] = sub_B2B5F0;
  sub_C53080(&qword_4F91460, "pgso-cutoff-sample-prof", 23);
  BYTE4(qword_4F914A8[10]) = 1;
  LODWORD(qword_4F914A8[8]) = 990000;
  unk_4F91490 = 79;
  LODWORD(qword_4F914A8[10]) = 990000;
  LOBYTE(word_4F9146C) = word_4F9146C & 0x9F | 0x20;
  unk_4F91488 = "The profile guided size optimization profile summary cutoff for sample profile.";
  sub_C53130(&qword_4F91460);
  return __cxa_atexit(sub_B2B680, &qword_4F91460, &qword_4A427C0);
}

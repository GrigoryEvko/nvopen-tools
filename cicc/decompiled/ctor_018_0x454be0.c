// Function: ctor_018
// Address: 0x454be0
//
int ctor_018()
{
  int v0; // edx
  __int64 v1; // r13
  __int64 v2; // rax
  int v3; // edx
  __int64 v4; // r15
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rax
  const char *v11; // [rsp+0h] [rbp-60h] BYREF
  char v12; // [rsp+20h] [rbp-40h]
  char v13; // [rsp+21h] [rbp-3Fh]

  qword_4F80F00 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F80F0C = word_4F80F0C & 0x8000;
  qword_4F80F48[1] = 0x100000000LL;
  unk_4F80F08 = v0;
  unk_4F80F10 = 0;
  unk_4F80F18 = 0;
  unk_4F80F20 = 0;
  unk_4F80F28 = 0;
  unk_4F80F30 = 0;
  unk_4F80F38 = 0;
  unk_4F80F40 = 0;
  qword_4F80F48[0] = &qword_4F80F48[2];
  qword_4F80F48[3] = 0;
  qword_4F80F48[4] = &qword_4F80F48[7];
  qword_4F80F48[5] = 1;
  LODWORD(qword_4F80F48[6]) = 0;
  BYTE4(qword_4F80F48[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_4F80F48[1]);
  if ( (unsigned __int64)LODWORD(qword_4F80F48[1]) + 1 > HIDWORD(qword_4F80F48[1]) )
  {
    sub_C8D5F0(qword_4F80F48, &qword_4F80F48[2], LODWORD(qword_4F80F48[1]) + 1LL, 8);
    v2 = LODWORD(qword_4F80F48[1]);
  }
  *(_QWORD *)(qword_4F80F48[0] + 8 * v2) = v1;
  qword_4F80F48[9] = &unk_49D9748;
  ++LODWORD(qword_4F80F48[1]);
  qword_4F80F48[8] = 0;
  qword_4F80F00 = &unk_49DC090;
  qword_4F80F48[10] = 0;
  qword_4F80F48[11] = &unk_49DC1D0;
  qword_4F80F48[15] = nullsub_23;
  qword_4F80F48[14] = sub_984030;
  sub_C53080(&qword_4F80F00, "experimental-debuginfo-iterators", 32);
  LOWORD(qword_4F80F48[10]) = 257;
  unk_4F80F28 = "Enable communicating debuginfo positions through iterators, eliminating intrinsics. Has no effect if --p"
                "reserve-input-debuginfo-format=true.";
  unk_4F80F30 = 140;
  LOBYTE(qword_4F80F48[8]) = 1;
  sub_C53130(&qword_4F80F00);
  __cxa_atexit(sub_984900, &qword_4F80F00, &qword_4A427C0);
  qword_4F80E20 = &unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F80E2C = word_4F80E2C & 0x8000;
  qword_4F80E68[1] = 0x100000000LL;
  unk_4F80E28 = v3;
  unk_4F80E30 = 0;
  unk_4F80E38 = 0;
  unk_4F80E40 = 0;
  unk_4F80E48 = 0;
  unk_4F80E50 = 0;
  unk_4F80E58 = 0;
  unk_4F80E60 = 0;
  qword_4F80E68[0] = &qword_4F80E68[2];
  qword_4F80E68[3] = 0;
  qword_4F80E68[4] = &qword_4F80E68[7];
  qword_4F80E68[5] = 1;
  LODWORD(qword_4F80E68[6]) = 0;
  BYTE4(qword_4F80E68[6]) = 1;
  v4 = sub_C57470();
  v5 = LODWORD(qword_4F80E68[1]);
  if ( (unsigned __int64)LODWORD(qword_4F80E68[1]) + 1 > HIDWORD(qword_4F80E68[1]) )
  {
    sub_C8D5F0(qword_4F80E68, &qword_4F80E68[2], LODWORD(qword_4F80E68[1]) + 1LL, 8);
    v5 = LODWORD(qword_4F80E68[1]);
  }
  *(_QWORD *)(qword_4F80E68[0] + 8 * v5) = v4;
  ++LODWORD(qword_4F80E68[1]);
  qword_4F80E68[8] = 0;
  qword_4F80E68[9] = &unk_49DC110;
  qword_4F80E68[10] = 0;
  qword_4F80E20 = &unk_49D97F0;
  qword_4F80E68[11] = &unk_49DC200;
  qword_4F80E68[15] = nullsub_26;
  qword_4F80E68[14] = sub_9C26D0;
  sub_C53080(&qword_4F80E20, "preserve-input-debuginfo-format", 31);
  unk_4F80E50 = 266;
  LOBYTE(word_4F80E2C) = word_4F80E2C & 0x9F | 0x20;
  unk_4F80E48 = "When set to true, IR files will be processed and printed in their current debug info format, regardless "
                "of default behaviour or other flags passed. Has no effect if input IR does not contain debug records or "
                "intrinsics. Ignored in llvm-link, llvm-lto, and llvm-lto2.";
  sub_C53130(&qword_4F80E20);
  __cxa_atexit(sub_9C44F0, &qword_4F80E20, &qword_4A427C0);
  qword_4F80D40 = &unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F80D4C = word_4F80D4C & 0x8000;
  unk_4F80D48 = v6;
  qword_4F80D88[1] = 0x100000000LL;
  unk_4F80D50 = 0;
  unk_4F80D58 = 0;
  unk_4F80D60 = 0;
  unk_4F80D68 = 0;
  unk_4F80D70 = 0;
  unk_4F80D78 = 0;
  unk_4F80D80 = 0;
  qword_4F80D88[0] = &qword_4F80D88[2];
  qword_4F80D88[3] = 0;
  qword_4F80D88[4] = &qword_4F80D88[7];
  qword_4F80D88[5] = 1;
  LODWORD(qword_4F80D88[6]) = 0;
  BYTE4(qword_4F80D88[6]) = 1;
  v7 = sub_C57470();
  v8 = LODWORD(qword_4F80D88[1]);
  if ( (unsigned __int64)LODWORD(qword_4F80D88[1]) + 1 > HIDWORD(qword_4F80D88[1]) )
  {
    sub_C8D5F0(qword_4F80D88, &qword_4F80D88[2], LODWORD(qword_4F80D88[1]) + 1LL, 8);
    v8 = LODWORD(qword_4F80D88[1]);
  }
  *(_QWORD *)(qword_4F80D88[0] + 8 * v8) = v7;
  qword_4F80D88[9] = &unk_49D9748;
  ++LODWORD(qword_4F80D88[1]);
  qword_4F80D88[8] = 0;
  qword_4F80D40 = &unk_49D9AD8;
  BYTE1(qword_4F80D88[10]) = 0;
  qword_4F80D88[11] = &unk_49DC1D0;
  qword_4F80D88[15] = nullsub_39;
  qword_4F80D88[14] = sub_AA4180;
  sub_C53080(&qword_4F80D40, "write-experimental-debuginfo-iterators-to-bitcode", 49);
  LOBYTE(word_4F80D4C) = word_4F80D4C & 0x9F | 0x20;
  if ( qword_4F80D88[8] )
  {
    v9 = sub_CEADF0();
    v13 = 1;
    v11 = "cl::location(x) specified more than once!";
    v12 = 3;
    sub_C53280(&qword_4F80D40, &v11, 0, 0, v9);
  }
  else
  {
    qword_4F80D88[8] = &unk_4F80E08;
  }
  *(_BYTE *)qword_4F80D88[8] = 1;
  LOWORD(qword_4F80D88[10]) = 257;
  sub_C53130(&qword_4F80D40);
  return __cxa_atexit(sub_AA4490, &qword_4F80D40, &qword_4A427C0);
}

// Function: ctor_383
// Address: 0x51a900
//
int ctor_383()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  int v3; // edx
  __int64 v4; // r15
  __int64 v5; // rax
  char v7; // [rsp+3h] [rbp-4Dh] BYREF
  int v8; // [rsp+4h] [rbp-4Ch] BYREF
  char *v9; // [rsp+8h] [rbp-48h] BYREF
  const char *v10; // [rsp+10h] [rbp-40h] BYREF
  __int64 v11; // [rsp+18h] [rbp-38h]

  qword_4FDBC00 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4FDBC0C = word_4FDBC0C & 0x8000;
  qword_4FDBC48[1] = 0x100000000LL;
  unk_4FDBC08 = v0;
  unk_4FDBC10 = 0;
  unk_4FDBC18 = 0;
  unk_4FDBC20 = 0;
  unk_4FDBC28 = 0;
  unk_4FDBC30 = 0;
  unk_4FDBC38 = 0;
  unk_4FDBC40 = 0;
  qword_4FDBC48[0] = &qword_4FDBC48[2];
  qword_4FDBC48[3] = 0;
  qword_4FDBC48[4] = &qword_4FDBC48[7];
  qword_4FDBC48[5] = 1;
  LODWORD(qword_4FDBC48[6]) = 0;
  BYTE4(qword_4FDBC48[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_4FDBC48[1]);
  if ( (unsigned __int64)LODWORD(qword_4FDBC48[1]) + 1 > HIDWORD(qword_4FDBC48[1]) )
  {
    sub_C8D5F0(qword_4FDBC48, &qword_4FDBC48[2], LODWORD(qword_4FDBC48[1]) + 1LL, 8);
    v2 = LODWORD(qword_4FDBC48[1]);
  }
  *(_QWORD *)(qword_4FDBC48[0] + 8 * v2) = v1;
  ++LODWORD(qword_4FDBC48[1]);
  qword_4FDBC48[8] = 0;
  qword_4FDBC48[9] = &unk_49D9748;
  qword_4FDBC48[10] = 0;
  qword_4FDBC00 = &unk_49DC090;
  qword_4FDBC48[11] = &unk_49DC1D0;
  qword_4FDBC48[15] = nullsub_23;
  qword_4FDBC48[14] = sub_984030;
  sub_C53080(&qword_4FDBC00, "no-ir-sim-branch-matching", 25);
  LOWORD(qword_4FDBC48[10]) = 256;
  LOBYTE(qword_4FDBC48[8]) = 0;
  unk_4FDBC30 = 83;
  LOBYTE(word_4FDBC0C) = word_4FDBC0C & 0x9F | 0x40;
  unk_4FDBC28 = "disable similarity matching, and outlining, across branches for debugging purposes.";
  sub_C53130(&qword_4FDBC00);
  __cxa_atexit(sub_984900, &qword_4FDBC00, &qword_4A427C0);
  qword_4FDBB20 = &unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4FDBB2C = word_4FDBB2C & 0x8000;
  qword_4FDBB68[1] = 0x100000000LL;
  unk_4FDBB28 = v3;
  unk_4FDBB30 = 0;
  unk_4FDBB38 = 0;
  unk_4FDBB40 = 0;
  unk_4FDBB48 = 0;
  unk_4FDBB50 = 0;
  unk_4FDBB58 = 0;
  unk_4FDBB60 = 0;
  qword_4FDBB68[0] = &qword_4FDBB68[2];
  qword_4FDBB68[3] = 0;
  qword_4FDBB68[4] = &qword_4FDBB68[7];
  qword_4FDBB68[5] = 1;
  LODWORD(qword_4FDBB68[6]) = 0;
  BYTE4(qword_4FDBB68[6]) = 1;
  v4 = sub_C57470();
  v5 = LODWORD(qword_4FDBB68[1]);
  if ( (unsigned __int64)LODWORD(qword_4FDBB68[1]) + 1 > HIDWORD(qword_4FDBB68[1]) )
  {
    sub_C8D5F0(qword_4FDBB68, &qword_4FDBB68[2], LODWORD(qword_4FDBB68[1]) + 1LL, 8);
    v5 = LODWORD(qword_4FDBB68[1]);
  }
  *(_QWORD *)(qword_4FDBB68[0] + 8 * v5) = v4;
  ++LODWORD(qword_4FDBB68[1]);
  qword_4FDBB68[8] = 0;
  qword_4FDBB68[9] = &unk_49D9748;
  qword_4FDBB68[10] = 0;
  qword_4FDBB20 = &unk_49DC090;
  qword_4FDBB68[11] = &unk_49DC1D0;
  qword_4FDBB68[15] = nullsub_23;
  qword_4FDBB68[14] = sub_984030;
  sub_C53080(&qword_4FDBB20, "no-ir-sim-indirect-calls", 24);
  LOBYTE(qword_4FDBB68[8]) = 0;
  LOWORD(qword_4FDBB68[10]) = 256;
  unk_4FDBB50 = 33;
  LOBYTE(word_4FDBB2C) = word_4FDBB2C & 0x9F | 0x40;
  unk_4FDBB48 = "disable outlining indirect calls.";
  sub_C53130(&qword_4FDBB20);
  __cxa_atexit(sub_984900, &qword_4FDBB20, &qword_4A427C0);
  v9 = &v7;
  v10 = "only allow matching call instructions if the name and type signature match.";
  v11 = 75;
  v8 = 2;
  v7 = 0;
  sub_22B0040(&unk_4FDBA40, "ir-sim-calls-by-name", &v9, &v8, &v10);
  __cxa_atexit(sub_984900, &unk_4FDBA40, &qword_4A427C0);
  v9 = &v7;
  v10 = "Don't match or outline intrinsics";
  v11 = 33;
  v8 = 2;
  v7 = 0;
  sub_22B0040(&unk_4FDB960, "no-ir-sim-intrinsics", &v9, &v8, &v10);
  return __cxa_atexit(sub_984900, &unk_4FDB960, &qword_4A427C0);
}

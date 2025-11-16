// Function: ctor_509
// Address: 0x55c980
//
int ctor_509()
{
  int v0; // edx
  __int64 v1; // r13
  __int64 v2; // rax
  int v3; // edx
  __int64 v4; // r13
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // rbx
  __int64 v8; // rax
  int v10; // [rsp+14h] [rbp-5Ch] BYREF
  int *v11; // [rsp+18h] [rbp-58h] BYREF
  const char *v12; // [rsp+20h] [rbp-50h] BYREF
  __int64 v13; // [rsp+28h] [rbp-48h]
  char *v14; // [rsp+30h] [rbp-40h] BYREF
  __int64 v15; // [rsp+38h] [rbp-38h]

  qword_500BD60 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_500BD6C = word_500BD6C & 0x8000;
  qword_500BDA8[1] = 0x100000000LL;
  unk_500BD68 = v0;
  unk_500BD70 = 0;
  unk_500BD78 = 0;
  unk_500BD80 = 0;
  unk_500BD88 = 0;
  unk_500BD90 = 0;
  unk_500BD98 = 0;
  unk_500BDA0 = 0;
  qword_500BDA8[0] = &qword_500BDA8[2];
  qword_500BDA8[3] = 0;
  qword_500BDA8[4] = &qword_500BDA8[7];
  qword_500BDA8[5] = 1;
  LODWORD(qword_500BDA8[6]) = 0;
  BYTE4(qword_500BDA8[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_500BDA8[1]);
  if ( (unsigned __int64)LODWORD(qword_500BDA8[1]) + 1 > HIDWORD(qword_500BDA8[1]) )
  {
    sub_C8D5F0(qword_500BDA8, &qword_500BDA8[2], LODWORD(qword_500BDA8[1]) + 1LL, 8);
    v2 = LODWORD(qword_500BDA8[1]);
  }
  *(_QWORD *)(qword_500BDA8[0] + 8 * v2) = v1;
  ++LODWORD(qword_500BDA8[1]);
  qword_500BDA8[8] = 0;
  qword_500BDA8[9] = &unk_49D9728;
  qword_500BDA8[10] = 0;
  qword_500BD60 = &unk_49DBF10;
  qword_500BDA8[11] = &unk_49DC290;
  qword_500BDA8[15] = nullsub_24;
  qword_500BDA8[14] = sub_984050;
  sub_C53080(&qword_500BD60, "sample-profile-max-propagate-iterations", 39);
  LODWORD(qword_500BDA8[8]) = 100;
  unk_500BD88 = "Maximum number of iterations to go through when propagating sample block/edge weights through the CFG.";
  BYTE4(qword_500BDA8[10]) = 1;
  LODWORD(qword_500BDA8[10]) = 100;
  unk_500BD90 = 102;
  sub_C53130(&qword_500BD60);
  __cxa_atexit(sub_984970, &qword_500BD60, &qword_4A427C0);
  v14 = "N";
  v11 = &v10;
  v12 = "Emit a warning if less than N% of records in the input profile are matched to the IR.";
  v13 = 85;
  v15 = 1;
  v10 = 0;
  sub_2A612E0(&unk_500BC80, "sample-profile-check-record-coverage", &v11, &v14, &v12);
  __cxa_atexit(sub_984970, &unk_500BC80, &qword_4A427C0);
  v14 = "N";
  v11 = &v10;
  v12 = "Emit a warning if less than N% of samples in the input profile are matched to the IR.";
  v13 = 85;
  v15 = 1;
  v10 = 0;
  sub_2A612E0(&unk_500BBA0, "sample-profile-check-sample-coverage", &v11, &v14, &v12);
  __cxa_atexit(sub_984970, &unk_500BBA0, &qword_4A427C0);
  qword_500BAC0 = &unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_500BACC = word_500BACC & 0x8000;
  qword_500BB08[1] = 0x100000000LL;
  unk_500BAC8 = v3;
  unk_500BAD0 = 0;
  unk_500BAD8 = 0;
  unk_500BAE0 = 0;
  unk_500BAE8 = 0;
  unk_500BAF0 = 0;
  unk_500BAF8 = 0;
  unk_500BB00 = 0;
  qword_500BB08[0] = &qword_500BB08[2];
  qword_500BB08[3] = 0;
  qword_500BB08[4] = &qword_500BB08[7];
  qword_500BB08[5] = 1;
  LODWORD(qword_500BB08[6]) = 0;
  BYTE4(qword_500BB08[6]) = 1;
  v4 = sub_C57470();
  v5 = LODWORD(qword_500BB08[1]);
  if ( (unsigned __int64)LODWORD(qword_500BB08[1]) + 1 > HIDWORD(qword_500BB08[1]) )
  {
    sub_C8D5F0(qword_500BB08, &qword_500BB08[2], LODWORD(qword_500BB08[1]) + 1LL, 8);
    v5 = LODWORD(qword_500BB08[1]);
  }
  *(_QWORD *)(qword_500BB08[0] + 8 * v5) = v4;
  qword_500BB08[9] = &unk_49D9748;
  ++LODWORD(qword_500BB08[1]);
  qword_500BB08[8] = 0;
  qword_500BAC0 = &unk_49DC090;
  qword_500BB08[10] = 0;
  qword_500BB08[11] = &unk_49DC1D0;
  qword_500BB08[15] = nullsub_23;
  qword_500BB08[14] = sub_984030;
  sub_C53080(&qword_500BAC0, "no-warn-sample-unused", 21);
  LOBYTE(qword_500BB08[8]) = 0;
  LOWORD(qword_500BB08[10]) = 256;
  unk_500BAF0 = 120;
  LOBYTE(word_500BACC) = word_500BACC & 0x9F | 0x20;
  unk_500BAE8 = "Use this option to turn off/on warnings about function with samples but without debug information to use"
                " those samples. ";
  sub_C53130(&qword_500BAC0);
  __cxa_atexit(sub_984900, &qword_500BAC0, &qword_4A427C0);
  qword_500B9E0 = &unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_500B9EC = word_500B9EC & 0x8000;
  unk_500B9E8 = v6;
  qword_500BA28[1] = 0x100000000LL;
  unk_500B9F0 = 0;
  unk_500B9F8 = 0;
  unk_500BA00 = 0;
  unk_500BA08 = 0;
  unk_500BA10 = 0;
  unk_500BA18 = 0;
  unk_500BA20 = 0;
  qword_500BA28[0] = &qword_500BA28[2];
  qword_500BA28[3] = 0;
  qword_500BA28[4] = &qword_500BA28[7];
  qword_500BA28[5] = 1;
  LODWORD(qword_500BA28[6]) = 0;
  BYTE4(qword_500BA28[6]) = 1;
  v7 = sub_C57470();
  v8 = LODWORD(qword_500BA28[1]);
  if ( (unsigned __int64)LODWORD(qword_500BA28[1]) + 1 > HIDWORD(qword_500BA28[1]) )
  {
    sub_C8D5F0(qword_500BA28, &qword_500BA28[2], LODWORD(qword_500BA28[1]) + 1LL, 8);
    v8 = LODWORD(qword_500BA28[1]);
  }
  *(_QWORD *)(qword_500BA28[0] + 8 * v8) = v7;
  qword_500BA28[9] = &unk_49D9748;
  ++LODWORD(qword_500BA28[1]);
  qword_500BA28[8] = 0;
  qword_500B9E0 = &unk_49DC090;
  qword_500BA28[10] = 0;
  qword_500BA28[11] = &unk_49DC1D0;
  qword_500BA28[15] = nullsub_23;
  qword_500BA28[14] = sub_984030;
  sub_C53080(&qword_500B9E0, "sample-profile-use-profi", 24);
  unk_500BA10 = 41;
  LOBYTE(word_500B9EC) = word_500B9EC & 0x9F | 0x20;
  unk_500BA08 = "Use profi to infer block and edge counts.";
  sub_C53130(&qword_500B9E0);
  return __cxa_atexit(sub_984900, &qword_500B9E0, &qword_4A427C0);
}

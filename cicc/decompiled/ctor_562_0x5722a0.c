// Function: ctor_562
// Address: 0x5722a0
//
int ctor_562()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  int v3; // edx
  __int64 v4; // r15
  __int64 v5; // rax

  qword_501F2C0 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_501F2CC = word_501F2CC & 0x8000;
  unk_501F2C8 = v0;
  qword_501F308[1] = 0x100000000LL;
  unk_501F2D0 = 0;
  unk_501F2D8 = 0;
  unk_501F2E0 = 0;
  unk_501F2E8 = 0;
  unk_501F2F0 = 0;
  unk_501F2F8 = 0;
  unk_501F300 = 0;
  qword_501F308[0] = &qword_501F308[2];
  qword_501F308[3] = 0;
  qword_501F308[4] = &qword_501F308[7];
  qword_501F308[5] = 1;
  LODWORD(qword_501F308[6]) = 0;
  BYTE4(qword_501F308[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_501F308[1]);
  if ( (unsigned __int64)LODWORD(qword_501F308[1]) + 1 > HIDWORD(qword_501F308[1]) )
  {
    sub_C8D5F0(qword_501F308, &qword_501F308[2], LODWORD(qword_501F308[1]) + 1LL, 8);
    v2 = LODWORD(qword_501F308[1]);
  }
  *(_QWORD *)(qword_501F308[0] + 8 * v2) = v1;
  ++LODWORD(qword_501F308[1]);
  qword_501F308[8] = 0;
  qword_501F308[9] = &unk_49D9728;
  qword_501F308[10] = 0;
  qword_501F2C0 = &unk_49DBF10;
  qword_501F308[11] = &unk_49DC290;
  qword_501F308[15] = nullsub_24;
  qword_501F308[14] = sub_984050;
  sub_C53080(&qword_501F2C0, "static-likely-prob", 18);
  unk_501F2F0 = 71;
  unk_501F2E8 = "branch probability threshold in percentage to be considered very likely";
  LODWORD(qword_501F308[8]) = 80;
  BYTE4(qword_501F308[10]) = 1;
  LODWORD(qword_501F308[10]) = 80;
  LOBYTE(word_501F2CC) = word_501F2CC & 0x9F | 0x20;
  sub_C53130(&qword_501F2C0);
  __cxa_atexit(sub_984970, &qword_501F2C0, &qword_4A427C0);
  qword_501F1E0 = &unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_501F1EC = word_501F1EC & 0x8000;
  unk_501F1F0 = 0;
  qword_501F228[1] = 0x100000000LL;
  unk_501F1E8 = v3;
  unk_501F1F8 = 0;
  unk_501F200 = 0;
  unk_501F208 = 0;
  unk_501F210 = 0;
  unk_501F218 = 0;
  unk_501F220 = 0;
  qword_501F228[0] = &qword_501F228[2];
  qword_501F228[3] = 0;
  qword_501F228[4] = &qword_501F228[7];
  qword_501F228[5] = 1;
  LODWORD(qword_501F228[6]) = 0;
  BYTE4(qword_501F228[6]) = 1;
  v4 = sub_C57470();
  v5 = LODWORD(qword_501F228[1]);
  if ( (unsigned __int64)LODWORD(qword_501F228[1]) + 1 > HIDWORD(qword_501F228[1]) )
  {
    sub_C8D5F0(qword_501F228, &qword_501F228[2], LODWORD(qword_501F228[1]) + 1LL, 8);
    v5 = LODWORD(qword_501F228[1]);
  }
  *(_QWORD *)(qword_501F228[0] + 8 * v5) = v4;
  ++LODWORD(qword_501F228[1]);
  qword_501F228[8] = 0;
  qword_501F228[9] = &unk_49D9728;
  qword_501F228[10] = 0;
  qword_501F1E0 = &unk_49DBF10;
  qword_501F228[11] = &unk_49DC290;
  qword_501F228[15] = nullsub_24;
  qword_501F228[14] = sub_984050;
  sub_C53080(&qword_501F1E0, "profile-likely-prob", 19);
  unk_501F210 = 97;
  unk_501F208 = "branch probability threshold in percentage to be considered very likely when profile is available";
  LODWORD(qword_501F228[8]) = 51;
  BYTE4(qword_501F228[10]) = 1;
  LODWORD(qword_501F228[10]) = 51;
  LOBYTE(word_501F1EC) = word_501F1EC & 0x9F | 0x20;
  sub_C53130(&qword_501F1E0);
  return __cxa_atexit(sub_984970, &qword_501F1E0, &qword_4A427C0);
}

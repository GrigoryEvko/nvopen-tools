// Function: ctor_518
// Address: 0x5620e0
//
int ctor_518()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  char *v8; // r13
  __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // rbx
  __int64 v12; // rax
  _QWORD v14[2]; // [rsp+10h] [rbp-50h] BYREF
  _BYTE v15[64]; // [rsp+20h] [rbp-40h] BYREF

  qword_5010880 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50108D0 = 0x100000000LL;
  dword_501088C &= 0x8000u;
  word_5010890 = 0;
  qword_5010898 = 0;
  qword_50108A0 = 0;
  dword_5010888 = v0;
  qword_50108A8 = 0;
  qword_50108B0 = 0;
  qword_50108B8 = 0;
  qword_50108C0 = 0;
  qword_50108C8 = (__int64)&unk_50108D8;
  qword_50108E0 = 0;
  qword_50108E8 = (__int64)&unk_5010900;
  qword_50108F0 = 1;
  dword_50108F8 = 0;
  byte_50108FC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_50108D0;
  v3 = (unsigned int)qword_50108D0 + 1LL;
  if ( v3 > HIDWORD(qword_50108D0) )
  {
    sub_C8D5F0((char *)&unk_50108D8 - 16, &unk_50108D8, v3, 8);
    v2 = (unsigned int)qword_50108D0;
  }
  *(_QWORD *)(qword_50108C8 + 8 * v2) = v1;
  LODWORD(qword_50108D0) = qword_50108D0 + 1;
  qword_5010908 = 0;
  qword_5010910 = (__int64)&unk_49D9748;
  qword_5010918 = 0;
  qword_5010880 = (__int64)&unk_49DC090;
  qword_5010920 = (__int64)&unk_49DC1D0;
  qword_5010940 = (__int64)nullsub_23;
  qword_5010938 = (__int64)sub_984030;
  sub_C53080(&qword_5010880, "sbvec-print-pass-pipeline", 25);
  LOWORD(qword_5010918) = 256;
  LOBYTE(qword_5010908) = 0;
  qword_50108B0 = 37;
  LOBYTE(dword_501088C) = dword_501088C & 0x9F | 0x20;
  qword_50108A8 = (__int64)"Prints the pass pipeline and returns.";
  sub_C53130(&qword_5010880);
  __cxa_atexit(sub_984900, &qword_5010880, &qword_4A427C0);
  qword_5010780 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50107D0 = 0x100000000LL;
  word_5010790 = 0;
  dword_501078C &= 0x8000u;
  qword_5010798 = 0;
  qword_50107A0 = 0;
  dword_5010788 = v4;
  qword_50107A8 = 0;
  qword_50107B0 = 0;
  qword_50107B8 = 0;
  qword_50107C0 = 0;
  qword_50107C8 = (__int64)&unk_50107D8;
  qword_50107E0 = 0;
  qword_50107E8 = (__int64)&unk_5010800;
  qword_50107F0 = 1;
  dword_50107F8 = 0;
  byte_50107FC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50107D0;
  v7 = (unsigned int)qword_50107D0 + 1LL;
  if ( v7 > HIDWORD(qword_50107D0) )
  {
    sub_C8D5F0((char *)&unk_50107D8 - 16, &unk_50107D8, v7, 8);
    v6 = (unsigned int)qword_50107D0;
  }
  *(_QWORD *)(qword_50107C8 + 8 * v6) = v5;
  qword_5010808 = (__int64)&byte_5010818;
  qword_5010830 = (__int64)&byte_5010840;
  LODWORD(qword_50107D0) = qword_50107D0 + 1;
  qword_5010810 = 0;
  qword_5010828 = (__int64)&unk_49DC130;
  byte_5010818 = 0;
  byte_5010840 = 0;
  qword_5010780 = (__int64)&unk_49DC010;
  qword_5010838 = 0;
  byte_5010850 = 0;
  qword_5010858 = (__int64)&unk_49DC350;
  qword_5010878 = (__int64)nullsub_92;
  qword_5010870 = (__int64)sub_BC4D70;
  sub_C53080(&qword_5010780, "sbvec-passes", 12);
  v8 = s;
  v14[0] = v15;
  v9 = -1;
  if ( s )
    v9 = (__int64)&v8[strlen(s)];
  sub_2BDC240(v14, v8, v9);
  sub_2240AE0(&qword_5010808, v14);
  byte_5010850 = 1;
  sub_2240AE0(&qword_5010830, v14);
  sub_2240A30(v14);
  qword_50107B0 = 85;
  LOBYTE(dword_501078C) = dword_501078C & 0x9F | 0x20;
  qword_50107A8 = (__int64)"Comma-separated list of vectorizer passes. If not set we run the predefined pipeline.";
  sub_C53130(&qword_5010780);
  __cxa_atexit(sub_BC5A40, &qword_5010780, &qword_4A427C0);
  qword_5010680 = &unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_501068C = word_501068C & 0x8000;
  unk_5010690 = 0;
  qword_50106C8[1] = 0x100000000LL;
  unk_5010688 = v10;
  qword_50106C8[0] = &qword_50106C8[2];
  unk_5010698 = 0;
  unk_50106A0 = 0;
  unk_50106A8 = 0;
  unk_50106B0 = 0;
  unk_50106B8 = 0;
  unk_50106C0 = 0;
  qword_50106C8[3] = 0;
  qword_50106C8[4] = &qword_50106C8[7];
  qword_50106C8[5] = 1;
  LODWORD(qword_50106C8[6]) = 0;
  BYTE4(qword_50106C8[6]) = 1;
  v11 = sub_C57470();
  v12 = LODWORD(qword_50106C8[1]);
  if ( (unsigned __int64)LODWORD(qword_50106C8[1]) + 1 > HIDWORD(qword_50106C8[1]) )
  {
    sub_C8D5F0(qword_50106C8, &qword_50106C8[2], LODWORD(qword_50106C8[1]) + 1LL, 8);
    v12 = LODWORD(qword_50106C8[1]);
  }
  *(_QWORD *)(qword_50106C8[0] + 8 * v12) = v11;
  qword_50106C8[8] = &qword_50106C8[10];
  qword_50106C8[13] = &qword_50106C8[15];
  ++LODWORD(qword_50106C8[1]);
  qword_50106C8[9] = 0;
  qword_50106C8[12] = &unk_49DC130;
  LOBYTE(qword_50106C8[10]) = 0;
  LOBYTE(qword_50106C8[15]) = 0;
  qword_5010680 = &unk_49DC010;
  qword_50106C8[14] = 0;
  LOBYTE(qword_50106C8[17]) = 0;
  qword_50106C8[18] = &unk_49DC350;
  qword_50106C8[22] = nullsub_92;
  qword_50106C8[21] = sub_BC4D70;
  sub_C53080(&qword_5010680, "sbvec-allow-files", 17);
  v14[0] = v15;
  sub_2BDC240(v14, ".*", "");
  sub_2240AE0(&qword_50106C8[8], v14);
  LOBYTE(qword_50106C8[17]) = 1;
  sub_2240AE0(&qword_50106C8[13], v14);
  sub_2240A30(v14);
  unk_50106B0 = 92;
  LOBYTE(word_501068C) = word_501068C & 0x9F | 0x20;
  unk_50106A8 = "Run the vectorizer only on file paths that match any in the list of comma-separated regex's.";
  sub_C53130(&qword_5010680);
  return __cxa_atexit(sub_BC5A40, &qword_5010680, &qword_4A427C0);
}

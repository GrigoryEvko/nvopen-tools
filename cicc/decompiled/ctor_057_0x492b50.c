// Function: ctor_057
// Address: 0x492b50
//
int ctor_057()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-38h]

  qword_4F87E40 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F87E90 = 0x100000000LL;
  dword_4F87E4C &= 0x8000u;
  word_4F87E50 = 0;
  qword_4F87E58 = 0;
  qword_4F87E60 = 0;
  dword_4F87E48 = v0;
  qword_4F87E68 = 0;
  qword_4F87E70 = 0;
  qword_4F87E78 = 0;
  qword_4F87E80 = 0;
  qword_4F87E88 = (__int64)&unk_4F87E98;
  qword_4F87EA0 = 0;
  qword_4F87EA8 = (__int64)&unk_4F87EC0;
  qword_4F87EB0 = 1;
  dword_4F87EB8 = 0;
  byte_4F87EBC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F87E90;
  v3 = (unsigned int)qword_4F87E90 + 1LL;
  if ( v3 > HIDWORD(qword_4F87E90) )
  {
    sub_C8D5F0((char *)&unk_4F87E98 - 16, &unk_4F87E98, v3, 8);
    v2 = (unsigned int)qword_4F87E90;
  }
  *(_QWORD *)(qword_4F87E88 + 8 * v2) = v1;
  qword_4F87ED0 = (__int64)&unk_49D9748;
  LODWORD(qword_4F87E90) = qword_4F87E90 + 1;
  qword_4F87EC8 = 0;
  qword_4F87E40 = (__int64)&unk_49DC090;
  qword_4F87ED8 = 0;
  qword_4F87EE0 = (__int64)&unk_49DC1D0;
  qword_4F87F00 = (__int64)nullsub_23;
  qword_4F87EF8 = (__int64)sub_984030;
  sub_C53080(&qword_4F87E40, "partial-profile", 15);
  LOWORD(qword_4F87ED8) = 256;
  LOBYTE(qword_4F87EC8) = 0;
  qword_4F87E70 = 57;
  LOBYTE(dword_4F87E4C) = dword_4F87E4C & 0x9F | 0x20;
  qword_4F87E68 = (__int64)"Specify the current profile is used as a partial profile.";
  sub_C53130(&qword_4F87E40);
  __cxa_atexit(sub_984900, &qword_4F87E40, &qword_4A427C0);
  qword_4F87D60 = &unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F87D6C = word_4F87D6C & 0x8000;
  qword_4F87DA8[1] = 0x100000000LL;
  unk_4F87D68 = v4;
  unk_4F87D70 = 0;
  unk_4F87D78 = 0;
  unk_4F87D80 = 0;
  unk_4F87D88 = 0;
  unk_4F87D90 = 0;
  unk_4F87D98 = 0;
  unk_4F87DA0 = 0;
  qword_4F87DA8[0] = &qword_4F87DA8[2];
  qword_4F87DA8[3] = 0;
  qword_4F87DA8[4] = &qword_4F87DA8[7];
  qword_4F87DA8[5] = 1;
  LODWORD(qword_4F87DA8[6]) = 0;
  BYTE4(qword_4F87DA8[6]) = 1;
  v5 = sub_C57470();
  v6 = LODWORD(qword_4F87DA8[1]);
  if ( (unsigned __int64)LODWORD(qword_4F87DA8[1]) + 1 > HIDWORD(qword_4F87DA8[1]) )
  {
    v12 = v5;
    sub_C8D5F0(qword_4F87DA8, &qword_4F87DA8[2], LODWORD(qword_4F87DA8[1]) + 1LL, 8);
    v6 = LODWORD(qword_4F87DA8[1]);
    v5 = v12;
  }
  *(_QWORD *)(qword_4F87DA8[0] + 8 * v6) = v5;
  qword_4F87DA8[9] = &unk_49D9748;
  ++LODWORD(qword_4F87DA8[1]);
  qword_4F87DA8[8] = 0;
  qword_4F87D60 = &unk_49DC090;
  qword_4F87DA8[10] = 0;
  qword_4F87DA8[11] = &unk_49DC1D0;
  qword_4F87DA8[15] = nullsub_23;
  qword_4F87DA8[14] = sub_984030;
  sub_C53080(&qword_4F87D60, "scale-partial-sample-profile-working-set-size", 45);
  LOBYTE(qword_4F87DA8[8]) = 1;
  unk_4F87D90 = 145;
  LOBYTE(word_4F87D6C) = word_4F87D6C & 0x9F | 0x20;
  LOWORD(qword_4F87DA8[10]) = 257;
  unk_4F87D88 = "If true, scale the working set size of the partial sample profile by the partial profile ratio to reflec"
                "t the size of the program being compiled.";
  sub_C53130(&qword_4F87D60);
  __cxa_atexit(sub_984900, &qword_4F87D60, &qword_4A427C0);
  qword_4F87C80 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F87CFC = 1;
  qword_4F87CD0 = 0x100000000LL;
  dword_4F87C8C &= 0x8000u;
  qword_4F87C98 = 0;
  qword_4F87CA0 = 0;
  qword_4F87CA8 = 0;
  dword_4F87C88 = v7;
  word_4F87C90 = 0;
  qword_4F87CB0 = 0;
  qword_4F87CB8 = 0;
  qword_4F87CC0 = 0;
  qword_4F87CC8 = (__int64)&unk_4F87CD8;
  qword_4F87CE0 = 0;
  qword_4F87CE8 = (__int64)&unk_4F87D00;
  qword_4F87CF0 = 1;
  dword_4F87CF8 = 0;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4F87CD0;
  v10 = (unsigned int)qword_4F87CD0 + 1LL;
  if ( v10 > HIDWORD(qword_4F87CD0) )
  {
    sub_C8D5F0((char *)&unk_4F87CD8 - 16, &unk_4F87CD8, v10, 8);
    v9 = (unsigned int)qword_4F87CD0;
  }
  *(_QWORD *)(qword_4F87CC8 + 8 * v9) = v8;
  LODWORD(qword_4F87CD0) = qword_4F87CD0 + 1;
  byte_4F87D20 = 0;
  qword_4F87D10 = (__int64)&unk_49DE5F0;
  qword_4F87D08 = 0;
  qword_4F87D18 = 0;
  qword_4F87C80 = (__int64)&unk_49DE610;
  qword_4F87D28 = (__int64)&unk_49DC2F0;
  qword_4F87D48 = (__int64)nullsub_190;
  qword_4F87D40 = (__int64)sub_D83E80;
  sub_C53080(&qword_4F87C80, "partial-sample-profile-working-set-size-scale-factor", 52);
  byte_4F87D20 = 1;
  qword_4F87D08 = 0x3F80624DD2F1A9FCLL;
  qword_4F87D18 = 0x3F80624DD2F1A9FCLL;
  LOBYTE(dword_4F87C8C) = dword_4F87C8C & 0x9F | 0x20;
  qword_4F87CA8 = (__int64)"The scale factor used to scale the working set size of the partial sample profile along with "
                           "the partial profile ratio. This includes the factor of the profile counter per block and the "
                           "factor to scale the working set size to use the same shared thresholds as PGO.";
  qword_4F87CB0 = 264;
  sub_C53130(&qword_4F87C80);
  return __cxa_atexit(sub_D84280, &qword_4F87C80, &qword_4A427C0);
}

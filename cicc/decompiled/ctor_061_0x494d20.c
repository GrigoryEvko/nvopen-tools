// Function: ctor_061
// Address: 0x494d20
//
int ctor_061()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // rbx
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 v16; // [rsp+8h] [rbp-38h]

  qword_4F89EE0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F89F30 = 0x100000000LL;
  dword_4F89EEC &= 0x8000u;
  word_4F89EF0 = 0;
  qword_4F89EF8 = 0;
  qword_4F89F00 = 0;
  dword_4F89EE8 = v0;
  qword_4F89F08 = 0;
  qword_4F89F10 = 0;
  qword_4F89F18 = 0;
  qword_4F89F20 = 0;
  qword_4F89F28 = (__int64)&unk_4F89F38;
  qword_4F89F40 = 0;
  qword_4F89F48 = (__int64)&unk_4F89F60;
  qword_4F89F50 = 1;
  dword_4F89F58 = 0;
  byte_4F89F5C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F89F30;
  v3 = (unsigned int)qword_4F89F30 + 1LL;
  if ( v3 > HIDWORD(qword_4F89F30) )
  {
    sub_C8D5F0((char *)&unk_4F89F38 - 16, &unk_4F89F38, v3, 8);
    v2 = (unsigned int)qword_4F89F30;
  }
  *(_QWORD *)(qword_4F89F28 + 8 * v2) = v1;
  LODWORD(qword_4F89F30) = qword_4F89F30 + 1;
  qword_4F89F68 = 0;
  qword_4F89F70 = (__int64)&unk_49D9748;
  qword_4F89F78 = 0;
  qword_4F89EE0 = (__int64)&unk_49DC090;
  qword_4F89F80 = (__int64)&unk_49DC1D0;
  qword_4F89FA0 = (__int64)nullsub_23;
  qword_4F89F98 = (__int64)sub_984030;
  sub_C53080(&qword_4F89EE0, "costmodel-reduxcost", 19);
  LOWORD(qword_4F89F78) = 256;
  LOBYTE(qword_4F89F68) = 0;
  qword_4F89F10 = 29;
  LOBYTE(dword_4F89EEC) = dword_4F89EEC & 0x9F | 0x20;
  qword_4F89F08 = (__int64)"Recognize reduction patterns.";
  sub_C53130(&qword_4F89EE0);
  __cxa_atexit(sub_984900, &qword_4F89EE0, &qword_4A427C0);
  qword_4F89E00 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F89E50 = 0x100000000LL;
  dword_4F89E0C &= 0x8000u;
  word_4F89E10 = 0;
  qword_4F89E18 = 0;
  qword_4F89E20 = 0;
  dword_4F89E08 = v4;
  qword_4F89E28 = 0;
  qword_4F89E30 = 0;
  qword_4F89E38 = 0;
  qword_4F89E40 = 0;
  qword_4F89E48 = (__int64)&unk_4F89E58;
  qword_4F89E60 = 0;
  qword_4F89E68 = (__int64)&unk_4F89E80;
  qword_4F89E70 = 1;
  dword_4F89E78 = 0;
  byte_4F89E7C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F89E50;
  v7 = (unsigned int)qword_4F89E50 + 1LL;
  if ( v7 > HIDWORD(qword_4F89E50) )
  {
    sub_C8D5F0((char *)&unk_4F89E58 - 16, &unk_4F89E58, v7, 8);
    v6 = (unsigned int)qword_4F89E50;
  }
  *(_QWORD *)(qword_4F89E48 + 8 * v6) = v5;
  qword_4F89E90 = (__int64)&unk_49D9728;
  qword_4F89E00 = (__int64)&unk_49DBF10;
  qword_4F89EA0 = (__int64)&unk_49DC290;
  LODWORD(qword_4F89E50) = qword_4F89E50 + 1;
  qword_4F89EC0 = (__int64)nullsub_24;
  qword_4F89E88 = 0;
  qword_4F89EB8 = (__int64)sub_984050;
  qword_4F89E98 = 0;
  sub_C53080(&qword_4F89E00, "cache-line-size", 15);
  LODWORD(qword_4F89E88) = 0;
  BYTE4(qword_4F89E98) = 1;
  LODWORD(qword_4F89E98) = 0;
  qword_4F89E30 = 75;
  LOBYTE(dword_4F89E0C) = dword_4F89E0C & 0x9F | 0x20;
  qword_4F89E28 = (__int64)"Use this to override the target cache line size when specified by the user.";
  sub_C53130(&qword_4F89E00);
  __cxa_atexit(sub_984970, &qword_4F89E00, &qword_4A427C0);
  qword_4F89D20 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F89D9C = 1;
  qword_4F89D70 = 0x100000000LL;
  dword_4F89D2C &= 0x8000u;
  qword_4F89D68 = (__int64)&unk_4F89D78;
  qword_4F89D38 = 0;
  qword_4F89D40 = 0;
  dword_4F89D28 = v8;
  word_4F89D30 = 0;
  qword_4F89D48 = 0;
  qword_4F89D50 = 0;
  qword_4F89D58 = 0;
  qword_4F89D60 = 0;
  qword_4F89D80 = 0;
  qword_4F89D88 = (__int64)&unk_4F89DA0;
  qword_4F89D90 = 1;
  dword_4F89D98 = 0;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4F89D70;
  if ( (unsigned __int64)(unsigned int)qword_4F89D70 + 1 > HIDWORD(qword_4F89D70) )
  {
    v16 = v9;
    sub_C8D5F0((char *)&unk_4F89D78 - 16, &unk_4F89D78, (unsigned int)qword_4F89D70 + 1LL, 8);
    v10 = (unsigned int)qword_4F89D70;
    v9 = v16;
  }
  *(_QWORD *)(qword_4F89D68 + 8 * v10) = v9;
  qword_4F89DB0 = (__int64)&unk_49D9728;
  qword_4F89D20 = (__int64)&unk_49DBF10;
  qword_4F89DC0 = (__int64)&unk_49DC290;
  LODWORD(qword_4F89D70) = qword_4F89D70 + 1;
  qword_4F89DE0 = (__int64)nullsub_24;
  qword_4F89DA8 = 0;
  qword_4F89DD8 = (__int64)sub_984050;
  qword_4F89DB8 = 0;
  sub_C53080(&qword_4F89D20, "min-page-size", 13);
  LODWORD(qword_4F89DA8) = 0;
  BYTE4(qword_4F89DB8) = 1;
  LODWORD(qword_4F89DB8) = 0;
  qword_4F89D50 = 52;
  LOBYTE(dword_4F89D2C) = dword_4F89D2C & 0x9F | 0x20;
  qword_4F89D48 = (__int64)"Use this to override the target's minimum page size.";
  sub_C53130(&qword_4F89D20);
  __cxa_atexit(sub_984970, &qword_4F89D20, &qword_4A427C0);
  qword_4F89C40 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F89C4C &= 0x8000u;
  word_4F89C50 = 0;
  qword_4F89C90 = 0x100000000LL;
  qword_4F89C88 = (__int64)&unk_4F89C98;
  qword_4F89C58 = 0;
  qword_4F89C60 = 0;
  dword_4F89C48 = v11;
  qword_4F89C68 = 0;
  qword_4F89C70 = 0;
  qword_4F89C78 = 0;
  qword_4F89C80 = 0;
  qword_4F89CA0 = 0;
  qword_4F89CA8 = (__int64)&unk_4F89CC0;
  qword_4F89CB0 = 1;
  dword_4F89CB8 = 0;
  byte_4F89CBC = 1;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_4F89C90;
  v14 = (unsigned int)qword_4F89C90 + 1LL;
  if ( v14 > HIDWORD(qword_4F89C90) )
  {
    sub_C8D5F0((char *)&unk_4F89C98 - 16, &unk_4F89C98, v14, 8);
    v13 = (unsigned int)qword_4F89C90;
  }
  *(_QWORD *)(qword_4F89C88 + 8 * v13) = v12;
  qword_4F89CD0 = (__int64)&unk_49D9728;
  qword_4F89C40 = (__int64)&unk_49DBF10;
  qword_4F89CE0 = (__int64)&unk_49DC290;
  LODWORD(qword_4F89C90) = qword_4F89C90 + 1;
  qword_4F89D00 = (__int64)nullsub_24;
  qword_4F89CC8 = 0;
  qword_4F89CF8 = (__int64)sub_984050;
  qword_4F89CD8 = 0;
  sub_C53080(&qword_4F89C40, "predictable-branch-threshold", 28);
  LODWORD(qword_4F89CC8) = 99;
  BYTE4(qword_4F89CD8) = 1;
  LODWORD(qword_4F89CD8) = 99;
  qword_4F89C70 = 67;
  LOBYTE(dword_4F89C4C) = dword_4F89C4C & 0x9F | 0x20;
  qword_4F89C68 = (__int64)"Use this to override the target's predictable branch threshold (%).";
  sub_C53130(&qword_4F89C40);
  return __cxa_atexit(sub_984970, &qword_4F89C40, &qword_4A427C0);
}

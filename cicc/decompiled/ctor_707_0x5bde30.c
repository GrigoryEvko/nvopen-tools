// Function: ctor_707
// Address: 0x5bde30
//
int __fastcall ctor_707(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r15
  __int64 v12; // rax
  unsigned __int64 v13; // rdx

  qword_5050EE0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5050F30 = 0x100000000LL;
  word_5050EF0 = 0;
  dword_5050EEC &= 0x8000u;
  qword_5050EF8 = 0;
  qword_5050F00 = 0;
  dword_5050EE8 = v4;
  qword_5050F08 = 0;
  qword_5050F10 = 0;
  qword_5050F18 = 0;
  qword_5050F20 = 0;
  qword_5050F28 = (__int64)&unk_5050F38;
  qword_5050F40 = 0;
  qword_5050F48 = (__int64)&unk_5050F60;
  qword_5050F50 = 1;
  dword_5050F58 = 0;
  byte_5050F5C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5050F30;
  v7 = (unsigned int)qword_5050F30 + 1LL;
  if ( v7 > HIDWORD(qword_5050F30) )
  {
    sub_C8D5F0((char *)&unk_5050F38 - 16, &unk_5050F38, v7, 8);
    v6 = (unsigned int)qword_5050F30;
  }
  *(_QWORD *)(qword_5050F28 + 8 * v6) = v5;
  qword_5050F70 = (__int64)&unk_49D9748;
  LODWORD(qword_5050F30) = qword_5050F30 + 1;
  qword_5050F68 = 0;
  qword_5050EE0 = (__int64)&unk_49DC090;
  qword_5050F80 = (__int64)&unk_49DC1D0;
  qword_5050F78 = 0;
  qword_5050FA0 = (__int64)nullsub_23;
  qword_5050F98 = (__int64)sub_984030;
  sub_C53080(&qword_5050EE0, "enable-legalize-types-checking", 30);
  LOBYTE(dword_5050EEC) = dword_5050EEC & 0x9F | 0x20;
  sub_C53130(&qword_5050EE0);
  __cxa_atexit(sub_984900, &qword_5050EE0, &qword_4A427C0);
  qword_5050E00 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5050EE0, v8, v9), 1u);
  byte_5050E7C = 1;
  qword_5050E50 = 0x100000000LL;
  dword_5050E0C &= 0x8000u;
  qword_5050E18 = 0;
  qword_5050E20 = 0;
  qword_5050E28 = 0;
  dword_5050E08 = v10;
  word_5050E10 = 0;
  qword_5050E30 = 0;
  qword_5050E38 = 0;
  qword_5050E40 = 0;
  qword_5050E48 = (__int64)&unk_5050E58;
  qword_5050E60 = 0;
  qword_5050E68 = (__int64)&unk_5050E80;
  qword_5050E70 = 1;
  dword_5050E78 = 0;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5050E50;
  v13 = (unsigned int)qword_5050E50 + 1LL;
  if ( v13 > HIDWORD(qword_5050E50) )
  {
    sub_C8D5F0((char *)&unk_5050E58 - 16, &unk_5050E58, v13, 8);
    v12 = (unsigned int)qword_5050E50;
  }
  *(_QWORD *)(qword_5050E48 + 8 * v12) = v11;
  qword_5050E90 = (__int64)&unk_49D9748;
  LODWORD(qword_5050E50) = qword_5050E50 + 1;
  qword_5050E88 = 0;
  qword_5050E00 = (__int64)&unk_49DC090;
  qword_5050EA0 = (__int64)&unk_49DC1D0;
  qword_5050E98 = 0;
  qword_5050EC0 = (__int64)nullsub_23;
  qword_5050EB8 = (__int64)sub_984030;
  sub_C53080(&qword_5050E00, "nvptx-generate-pack-unpack", 26);
  LOBYTE(qword_5050E88) = 1;
  qword_5050E30 = 60;
  LOBYTE(dword_5050E0C) = dword_5050E0C & 0x9F | 0x20;
  LOWORD(qword_5050E98) = 257;
  qword_5050E28 = (__int64)"Generate packing/unpack moves in place of bitwise operations";
  sub_C53130(&qword_5050E00);
  return __cxa_atexit(sub_984900, &qword_5050E00, &qword_4A427C0);
}

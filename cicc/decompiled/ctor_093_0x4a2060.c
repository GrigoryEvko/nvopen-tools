// Function: ctor_093
// Address: 0x4a2060
//
__int64 ctor_093()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  int v3; // edx
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 result; // rax
  _QWORD v9[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v10[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v11[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v12[8]; // [rsp+30h] [rbp-40h] BYREF

  qword_4F90D60 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F90D6C = word_4F90D6C & 0x8000;
  qword_4F90DA8[1] = 0x100000000LL;
  unk_4F90D68 = v0;
  unk_4F90D70 = 0;
  unk_4F90D78 = 0;
  unk_4F90D80 = 0;
  unk_4F90D88 = 0;
  unk_4F90D90 = 0;
  unk_4F90D98 = 0;
  unk_4F90DA0 = 0;
  qword_4F90DA8[0] = &qword_4F90DA8[2];
  qword_4F90DA8[3] = 0;
  qword_4F90DA8[4] = &qword_4F90DA8[7];
  qword_4F90DA8[5] = 1;
  LODWORD(qword_4F90DA8[6]) = 0;
  BYTE4(qword_4F90DA8[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_4F90DA8[1]);
  if ( (unsigned __int64)LODWORD(qword_4F90DA8[1]) + 1 > HIDWORD(qword_4F90DA8[1]) )
  {
    sub_C8D5F0(qword_4F90DA8, &qword_4F90DA8[2], LODWORD(qword_4F90DA8[1]) + 1LL, 8);
    v2 = LODWORD(qword_4F90DA8[1]);
  }
  *(_QWORD *)(qword_4F90DA8[0] + 8 * v2) = v1;
  ++LODWORD(qword_4F90DA8[1]);
  qword_4F90DA8[8] = 0;
  qword_4F90DA8[9] = &unk_49D9748;
  qword_4F90DA8[10] = 0;
  qword_4F90D60 = &unk_49DC090;
  qword_4F90DA8[11] = &unk_49DC1D0;
  qword_4F90DA8[15] = nullsub_23;
  qword_4F90DA8[14] = sub_984030;
  sub_C53080(&qword_4F90D60, "assume-preserve-all", 19);
  LOWORD(qword_4F90DA8[10]) = 256;
  LOBYTE(qword_4F90DA8[8]) = 0;
  unk_4F90D90 = 80;
  LOBYTE(word_4F90D6C) = word_4F90D6C & 0x9F | 0x20;
  unk_4F90D88 = "enable preservation of all attributes. even those that are unlikely to be useful";
  sub_C53130(&qword_4F90D60);
  __cxa_atexit(sub_984900, &qword_4F90D60, &qword_4A427C0);
  qword_4F90C80 = &unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F90C8C = word_4F90C8C & 0x8000;
  qword_4F90CC8[1] = 0x100000000LL;
  unk_4F90C88 = v3;
  unk_4F90C90 = 0;
  unk_4F90C98 = 0;
  unk_4F90CA0 = 0;
  unk_4F90CA8 = 0;
  unk_4F90CB0 = 0;
  unk_4F90CB8 = 0;
  unk_4F90CC0 = 0;
  qword_4F90CC8[0] = &qword_4F90CC8[2];
  qword_4F90CC8[3] = 0;
  qword_4F90CC8[4] = &qword_4F90CC8[7];
  qword_4F90CC8[5] = 1;
  LODWORD(qword_4F90CC8[6]) = 0;
  BYTE4(qword_4F90CC8[6]) = 1;
  v4 = sub_C57470();
  v5 = LODWORD(qword_4F90CC8[1]);
  if ( (unsigned __int64)LODWORD(qword_4F90CC8[1]) + 1 > HIDWORD(qword_4F90CC8[1]) )
  {
    sub_C8D5F0(qword_4F90CC8, &qword_4F90CC8[2], LODWORD(qword_4F90CC8[1]) + 1LL, 8);
    v5 = LODWORD(qword_4F90CC8[1]);
  }
  *(_QWORD *)(qword_4F90CC8[0] + 8 * v5) = v4;
  ++LODWORD(qword_4F90CC8[1]);
  qword_4F90CC8[8] = 0;
  qword_4F90CC8[9] = &unk_49D9748;
  qword_4F90CC8[10] = 0;
  qword_4F90C80 = &unk_49DC090;
  qword_4F90CC8[11] = &unk_49DC1D0;
  qword_4F90CC8[15] = nullsub_23;
  qword_4F90CC8[14] = sub_984030;
  sub_C53080(&qword_4F90C80, "enable-knowledge-retention", 26);
  LOBYTE(qword_4F90CC8[8]) = 0;
  LOWORD(qword_4F90CC8[10]) = 256;
  unk_4F90CB0 = 64;
  LOBYTE(word_4F90C8C) = word_4F90C8C & 0x9F | 0x20;
  unk_4F90CA8 = "enable preservation of attributes throughout code transformation";
  sub_C53130(&qword_4F90C80);
  __cxa_atexit(sub_984900, &qword_4F90C80, &qword_4A427C0);
  v6 = sub_C60B10();
  v11[0] = v12;
  v7 = v6;
  sub_11BE0C0(v11, (char *)&unk_3F704E3 - 35);
  v9[0] = v10;
  sub_11BE0C0(v9, "assume-builder-counter");
  result = sub_CF9810(v7, v9, v11);
  if ( (_QWORD *)v9[0] != v10 )
    result = j_j___libc_free_0(v9[0], v10[0] + 1LL);
  if ( (_QWORD *)v11[0] != v12 )
    return j_j___libc_free_0(v11[0], v12[0] + 1LL);
  return result;
}

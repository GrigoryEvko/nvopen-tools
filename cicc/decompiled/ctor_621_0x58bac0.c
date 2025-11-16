// Function: ctor_621
// Address: 0x58bac0
//
int __fastcall ctor_621(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
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

  qword_502EAE0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_502EB30 = 0x100000000LL;
  dword_502EAEC &= 0x8000u;
  word_502EAF0 = 0;
  qword_502EAF8 = 0;
  qword_502EB00 = 0;
  dword_502EAE8 = v4;
  qword_502EB08 = 0;
  qword_502EB10 = 0;
  qword_502EB18 = 0;
  qword_502EB20 = 0;
  qword_502EB28 = (__int64)&unk_502EB38;
  qword_502EB40 = 0;
  qword_502EB48 = (__int64)&unk_502EB60;
  qword_502EB50 = 1;
  dword_502EB58 = 0;
  byte_502EB5C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_502EB30;
  v7 = (unsigned int)qword_502EB30 + 1LL;
  if ( v7 > HIDWORD(qword_502EB30) )
  {
    sub_C8D5F0((char *)&unk_502EB38 - 16, &unk_502EB38, v7, 8);
    v6 = (unsigned int)qword_502EB30;
  }
  *(_QWORD *)(qword_502EB28 + 8 * v6) = v5;
  qword_502EB70 = (__int64)&unk_49D9748;
  LODWORD(qword_502EB30) = qword_502EB30 + 1;
  qword_502EB68 = 0;
  qword_502EAE0 = (__int64)&unk_49DC090;
  qword_502EB80 = (__int64)&unk_49DC1D0;
  qword_502EB78 = 0;
  qword_502EBA0 = (__int64)nullsub_23;
  qword_502EB98 = (__int64)sub_984030;
  sub_C53080(&qword_502EAE0, "ddg-simplify", 12);
  LOWORD(qword_502EB78) = 257;
  LOBYTE(qword_502EB68) = 1;
  qword_502EB10 = 63;
  LOBYTE(dword_502EAEC) = dword_502EAEC & 0x9F | 0x20;
  qword_502EB08 = (__int64)"Simplify DDG by merging nodes that have less interesting edges.";
  sub_C53130(&qword_502EAE0);
  __cxa_atexit(sub_984900, &qword_502EAE0, &qword_4A427C0);
  qword_502EA00 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502EAE0, v8, v9), 1u);
  qword_502EA50 = 0x100000000LL;
  word_502EA10 = 0;
  dword_502EA0C &= 0x8000u;
  qword_502EA18 = 0;
  qword_502EA20 = 0;
  dword_502EA08 = v10;
  qword_502EA28 = 0;
  qword_502EA30 = 0;
  qword_502EA38 = 0;
  qword_502EA40 = 0;
  qword_502EA48 = (__int64)&unk_502EA58;
  qword_502EA60 = 0;
  qword_502EA68 = (__int64)&unk_502EA80;
  qword_502EA70 = 1;
  dword_502EA78 = 0;
  byte_502EA7C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_502EA50;
  v13 = (unsigned int)qword_502EA50 + 1LL;
  if ( v13 > HIDWORD(qword_502EA50) )
  {
    sub_C8D5F0((char *)&unk_502EA58 - 16, &unk_502EA58, v13, 8);
    v12 = (unsigned int)qword_502EA50;
  }
  *(_QWORD *)(qword_502EA48 + 8 * v12) = v11;
  qword_502EA90 = (__int64)&unk_49D9748;
  LODWORD(qword_502EA50) = qword_502EA50 + 1;
  qword_502EA88 = 0;
  qword_502EA00 = (__int64)&unk_49DC090;
  qword_502EAA0 = (__int64)&unk_49DC1D0;
  qword_502EA98 = 0;
  qword_502EAC0 = (__int64)nullsub_23;
  qword_502EAB8 = (__int64)sub_984030;
  sub_C53080(&qword_502EA00, "ddg-pi-blocks", 13);
  LOBYTE(qword_502EA88) = 1;
  LOWORD(qword_502EA98) = 257;
  qword_502EA30 = 22;
  LOBYTE(dword_502EA0C) = dword_502EA0C & 0x9F | 0x20;
  qword_502EA28 = (__int64)"Create pi-block nodes.";
  sub_C53130(&qword_502EA00);
  return __cxa_atexit(sub_984900, &qword_502EA00, &qword_4A427C0);
}

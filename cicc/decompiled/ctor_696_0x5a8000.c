// Function: ctor_696
// Address: 0x5a8000
//
int __fastcall ctor_696(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5040FA0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_504101C = 1;
  qword_5040FF0 = 0x100000000LL;
  dword_5040FAC &= 0x8000u;
  qword_5040FB8 = 0;
  qword_5040FC0 = 0;
  qword_5040FC8 = 0;
  dword_5040FA8 = v4;
  word_5040FB0 = 0;
  qword_5040FD0 = 0;
  qword_5040FD8 = 0;
  qword_5040FE0 = 0;
  qword_5040FE8 = (__int64)&unk_5040FF8;
  qword_5041000 = 0;
  qword_5041008 = (__int64)&unk_5041020;
  qword_5041010 = 1;
  dword_5041018 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5040FF0;
  v7 = (unsigned int)qword_5040FF0 + 1LL;
  if ( v7 > HIDWORD(qword_5040FF0) )
  {
    sub_C8D5F0((char *)&unk_5040FF8 - 16, &unk_5040FF8, v7, 8);
    v6 = (unsigned int)qword_5040FF0;
  }
  *(_QWORD *)(qword_5040FE8 + 8 * v6) = v5;
  LODWORD(qword_5040FF0) = qword_5040FF0 + 1;
  qword_5041028 = 0;
  qword_5041030 = (__int64)&unk_49D9748;
  qword_5041038 = 0;
  qword_5040FA0 = (__int64)&unk_49DC090;
  qword_5041040 = (__int64)&unk_49DC1D0;
  qword_5041060 = (__int64)nullsub_23;
  qword_5041058 = (__int64)sub_984030;
  sub_C53080(&qword_5040FA0, "vasp-fix2", 9);
  LOBYTE(qword_5041028) = 0;
  LOWORD(qword_5041038) = 256;
  qword_5040FD0 = 0;
  LOBYTE(dword_5040FAC) = dword_5040FAC & 0x9F | 0x20;
  qword_5040FC8 = (__int64)byte_3F871B3;
  sub_C53130(&qword_5040FA0);
  return __cxa_atexit(sub_984900, &qword_5040FA0, &qword_4A427C0);
}

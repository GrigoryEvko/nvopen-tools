// Function: ctor_601
// Address: 0x583910
//
int __fastcall ctor_601(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_502A120 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  dword_502A12C &= 0x8000u;
  word_502A130 = 0;
  qword_502A170 = 0x100000000LL;
  qword_502A138 = 0;
  qword_502A140 = 0;
  qword_502A148 = 0;
  dword_502A128 = v4;
  qword_502A150 = 0;
  qword_502A158 = 0;
  qword_502A160 = 0;
  qword_502A168 = (__int64)&unk_502A178;
  qword_502A180 = 0;
  qword_502A188 = (__int64)&unk_502A1A0;
  qword_502A190 = 1;
  dword_502A198 = 0;
  byte_502A19C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_502A170;
  v7 = (unsigned int)qword_502A170 + 1LL;
  if ( v7 > HIDWORD(qword_502A170) )
  {
    sub_C8D5F0((char *)&unk_502A178 - 16, &unk_502A178, v7, 8);
    v6 = (unsigned int)qword_502A170;
  }
  *(_QWORD *)(qword_502A168 + 8 * v6) = v5;
  LODWORD(qword_502A170) = qword_502A170 + 1;
  qword_502A1A8 = 0;
  qword_502A1B0 = (__int64)&unk_49D9728;
  qword_502A1B8 = 0;
  qword_502A120 = (__int64)&unk_49DBF10;
  qword_502A1C0 = (__int64)&unk_49DC290;
  qword_502A1E0 = (__int64)nullsub_24;
  qword_502A1D8 = (__int64)sub_984050;
  sub_C53080(&qword_502A120, "huge-size-for-split", 19);
  qword_502A150 = 90;
  LODWORD(qword_502A1A8) = 5000;
  BYTE4(qword_502A1B8) = 1;
  LODWORD(qword_502A1B8) = 5000;
  LOBYTE(dword_502A12C) = dword_502A12C & 0x9F | 0x20;
  qword_502A148 = (__int64)"A threshold of live range size which may cause high compile time cost in global splitting.";
  sub_C53130(&qword_502A120);
  return __cxa_atexit(sub_984970, &qword_502A120, &qword_4A427C0);
}

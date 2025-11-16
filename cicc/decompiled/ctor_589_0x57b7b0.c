// Function: ctor_589
// Address: 0x57b7b0
//
int __fastcall ctor_589(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_50251A0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_502521C = 1;
  qword_50251F0 = 0x100000000LL;
  dword_50251AC &= 0x8000u;
  qword_50251B8 = 0;
  qword_50251C0 = 0;
  qword_50251C8 = 0;
  dword_50251A8 = v4;
  word_50251B0 = 0;
  qword_50251D0 = 0;
  qword_50251D8 = 0;
  qword_50251E0 = 0;
  qword_50251E8 = (__int64)&unk_50251F8;
  qword_5025200 = 0;
  qword_5025208 = (__int64)&unk_5025220;
  qword_5025210 = 1;
  dword_5025218 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50251F0;
  v7 = (unsigned int)qword_50251F0 + 1LL;
  if ( v7 > HIDWORD(qword_50251F0) )
  {
    sub_C8D5F0((char *)&unk_50251F8 - 16, &unk_50251F8, v7, 8);
    v6 = (unsigned int)qword_50251F0;
  }
  *(_QWORD *)(qword_50251E8 + 8 * v6) = v5;
  LODWORD(qword_50251F0) = qword_50251F0 + 1;
  qword_5025228 = 0;
  qword_5025230 = (__int64)&unk_49D9748;
  qword_5025238 = 0;
  qword_50251A0 = (__int64)&unk_49DC090;
  qword_5025240 = (__int64)&unk_49DC1D0;
  qword_5025260 = (__int64)nullsub_23;
  qword_5025258 = (__int64)sub_984030;
  sub_C53080(&qword_50251A0, "safe-stack-layout", 17);
  qword_50251D0 = 24;
  qword_50251C8 = (__int64)"enable safe stack layout";
  LOBYTE(qword_5025228) = 1;
  LOBYTE(dword_50251AC) = dword_50251AC & 0x9F | 0x20;
  LOWORD(qword_5025238) = 257;
  sub_C53130(&qword_50251A0);
  return __cxa_atexit(sub_984900, &qword_50251A0, &qword_4A427C0);
}

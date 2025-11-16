// Function: ctor_598
// Address: 0x57e5e0
//
int __fastcall ctor_598(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5026960 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_50269DC = 1;
  qword_50269B0 = 0x100000000LL;
  dword_502696C &= 0x8000u;
  qword_5026978 = 0;
  qword_5026980 = 0;
  qword_5026988 = 0;
  dword_5026968 = v4;
  word_5026970 = 0;
  qword_5026990 = 0;
  qword_5026998 = 0;
  qword_50269A0 = 0;
  qword_50269A8 = (__int64)&unk_50269B8;
  qword_50269C0 = 0;
  qword_50269C8 = (__int64)&unk_50269E0;
  qword_50269D0 = 1;
  dword_50269D8 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50269B0;
  v7 = (unsigned int)qword_50269B0 + 1LL;
  if ( v7 > HIDWORD(qword_50269B0) )
  {
    sub_C8D5F0((char *)&unk_50269B8 - 16, &unk_50269B8, v7, 8);
    v6 = (unsigned int)qword_50269B0;
  }
  *(_QWORD *)(qword_50269A8 + 8 * v6) = v5;
  LODWORD(qword_50269B0) = qword_50269B0 + 1;
  qword_50269E8 = 0;
  qword_50269F0 = (__int64)&unk_49D9748;
  qword_50269F8 = 0;
  qword_5026960 = (__int64)&unk_49DC090;
  qword_5026A00 = (__int64)&unk_49DC1D0;
  qword_5026A20 = (__int64)nullsub_23;
  qword_5026A18 = (__int64)sub_984030;
  sub_C53080(&qword_5026960, "disable-sched-hazard", 20);
  LOBYTE(qword_50269E8) = 0;
  qword_5026990 = 48;
  LOBYTE(dword_502696C) = dword_502696C & 0x9F | 0x20;
  LOWORD(qword_50269F8) = 256;
  qword_5026988 = (__int64)"Disable hazard detection during preRA scheduling";
  sub_C53130(&qword_5026960);
  return __cxa_atexit(sub_984900, &qword_5026960, &qword_4A427C0);
}

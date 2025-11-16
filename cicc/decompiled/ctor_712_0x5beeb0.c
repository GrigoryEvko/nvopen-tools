// Function: ctor_712
// Address: 0x5beeb0
//
int __fastcall ctor_712(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5051520 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  dword_505152C &= 0x8000u;
  word_5051530 = 0;
  qword_5051570 = 0x100000000LL;
  qword_5051538 = 0;
  qword_5051540 = 0;
  qword_5051548 = 0;
  dword_5051528 = v4;
  qword_5051550 = 0;
  qword_5051558 = 0;
  qword_5051560 = 0;
  qword_5051568 = (__int64)&unk_5051578;
  qword_5051580 = 0;
  qword_5051588 = (__int64)&unk_50515A0;
  qword_5051590 = 1;
  dword_5051598 = 0;
  byte_505159C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5051570;
  v7 = (unsigned int)qword_5051570 + 1LL;
  if ( v7 > HIDWORD(qword_5051570) )
  {
    sub_C8D5F0((char *)&unk_5051578 - 16, &unk_5051578, v7, 8);
    v6 = (unsigned int)qword_5051570;
  }
  *(_QWORD *)(qword_5051568 + 8 * v6) = v5;
  LODWORD(qword_5051570) = qword_5051570 + 1;
  qword_50515A8 = 0;
  qword_50515B0 = (__int64)&unk_49D9748;
  qword_50515B8 = 0;
  qword_5051520 = (__int64)&unk_49DC090;
  qword_50515C0 = (__int64)&unk_49DC1D0;
  qword_50515E0 = (__int64)nullsub_23;
  qword_50515D8 = (__int64)sub_984030;
  sub_C53080(&qword_5051520, "print-all-reaching-defs", 23);
  qword_5051550 = 22;
  LOBYTE(dword_505152C) = dword_505152C & 0x9F | 0x20;
  qword_5051548 = (__int64)"Used for test purpuses";
  sub_C53130(&qword_5051520);
  return __cxa_atexit(sub_984900, &qword_5051520, &qword_4A427C0);
}

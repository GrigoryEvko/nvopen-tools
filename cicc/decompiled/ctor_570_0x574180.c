// Function: ctor_570
// Address: 0x574180
//
int ctor_570()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_50208E0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50208EC &= 0x8000u;
  word_50208F0 = 0;
  qword_5020930 = 0x100000000LL;
  qword_50208F8 = 0;
  qword_5020900 = 0;
  qword_5020908 = 0;
  dword_50208E8 = v0;
  qword_5020910 = 0;
  qword_5020918 = 0;
  qword_5020920 = 0;
  qword_5020928 = (__int64)&unk_5020938;
  qword_5020940 = 0;
  qword_5020948 = (__int64)&unk_5020960;
  qword_5020950 = 1;
  dword_5020958 = 0;
  byte_502095C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5020930;
  v3 = (unsigned int)qword_5020930 + 1LL;
  if ( v3 > HIDWORD(qword_5020930) )
  {
    sub_C8D5F0((char *)&unk_5020938 - 16, &unk_5020938, v3, 8);
    v2 = (unsigned int)qword_5020930;
  }
  *(_QWORD *)(qword_5020928 + 8 * v2) = v1;
  LODWORD(qword_5020930) = qword_5020930 + 1;
  qword_5020968 = 0;
  qword_5020970 = (__int64)&unk_49DA090;
  qword_5020978 = 0;
  qword_50208E0 = (__int64)&unk_49DBF90;
  qword_5020980 = (__int64)&unk_49DC230;
  qword_50209A0 = (__int64)nullsub_58;
  qword_5020998 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_50208E0, "print-regmask-num-regs", 22);
  qword_5020910 = 90;
  qword_5020908 = (__int64)"Number of registers to limit to when printing regmask operands in IR dumps. unlimited = -1";
  LODWORD(qword_5020968) = 32;
  BYTE4(qword_5020978) = 1;
  LODWORD(qword_5020978) = 32;
  LOBYTE(dword_50208EC) = dword_50208EC & 0x9F | 0x20;
  sub_C53130(&qword_50208E0);
  return __cxa_atexit(sub_B2B680, &qword_50208E0, &qword_4A427C0);
}

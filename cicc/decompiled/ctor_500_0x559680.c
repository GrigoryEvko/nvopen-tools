// Function: ctor_500
// Address: 0x559680
//
int ctor_500()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_5009F00 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5009F7C = 1;
  qword_5009F50 = 0x100000000LL;
  dword_5009F0C &= 0x8000u;
  qword_5009F18 = 0;
  qword_5009F20 = 0;
  qword_5009F28 = 0;
  dword_5009F08 = v0;
  word_5009F10 = 0;
  qword_5009F30 = 0;
  qword_5009F38 = 0;
  qword_5009F40 = 0;
  qword_5009F48 = (__int64)&unk_5009F58;
  qword_5009F60 = 0;
  qword_5009F68 = (__int64)&unk_5009F80;
  qword_5009F70 = 1;
  dword_5009F78 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5009F50;
  v3 = (unsigned int)qword_5009F50 + 1LL;
  if ( v3 > HIDWORD(qword_5009F50) )
  {
    sub_C8D5F0((char *)&unk_5009F58 - 16, &unk_5009F58, v3, 8);
    v2 = (unsigned int)qword_5009F50;
  }
  *(_QWORD *)(qword_5009F48 + 8 * v2) = v1;
  LODWORD(qword_5009F50) = qword_5009F50 + 1;
  qword_5009F88 = 0;
  qword_5009F90 = (__int64)&unk_49D9748;
  qword_5009F98 = 0;
  qword_5009F00 = (__int64)&unk_49DC090;
  qword_5009FA0 = (__int64)&unk_49DC1D0;
  qword_5009FC0 = (__int64)nullsub_23;
  qword_5009FB8 = (__int64)sub_984030;
  sub_C53080(&qword_5009F00, "loop-rotate-multi", 17);
  LOBYTE(qword_5009F88) = 0;
  LOWORD(qword_5009F98) = 256;
  qword_5009F30 = 72;
  LOBYTE(dword_5009F0C) = dword_5009F0C & 0x9F | 0x20;
  qword_5009F28 = (__int64)"Allow loop rotation multiple times in order to reach a better latch exit";
  sub_C53130(&qword_5009F00);
  return __cxa_atexit(sub_984900, &qword_5009F00, &qword_4A427C0);
}

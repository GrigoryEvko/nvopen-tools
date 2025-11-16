// Function: ctor_684
// Address: 0x5a5f60
//
int __fastcall ctor_684(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_503FC20 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_503FC9C = 1;
  qword_503FC70 = 0x100000000LL;
  dword_503FC2C &= 0x8000u;
  qword_503FC38 = 0;
  qword_503FC40 = 0;
  qword_503FC48 = 0;
  dword_503FC28 = v4;
  word_503FC30 = 0;
  qword_503FC50 = 0;
  qword_503FC58 = 0;
  qword_503FC60 = 0;
  qword_503FC68 = (__int64)&unk_503FC78;
  qword_503FC80 = 0;
  qword_503FC88 = (__int64)&unk_503FCA0;
  qword_503FC90 = 1;
  dword_503FC98 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_503FC70;
  v7 = (unsigned int)qword_503FC70 + 1LL;
  if ( v7 > HIDWORD(qword_503FC70) )
  {
    sub_C8D5F0((char *)&unk_503FC78 - 16, &unk_503FC78, v7, 8);
    v6 = (unsigned int)qword_503FC70;
  }
  *(_QWORD *)(qword_503FC68 + 8 * v6) = v5;
  LODWORD(qword_503FC70) = qword_503FC70 + 1;
  qword_503FCA8 = 0;
  qword_503FCB0 = (__int64)&unk_49D9748;
  qword_503FCB8 = 0;
  qword_503FC20 = (__int64)&unk_49DC090;
  qword_503FCC0 = (__int64)&unk_49DC1D0;
  qword_503FCE0 = (__int64)nullsub_23;
  qword_503FCD8 = (__int64)sub_984030;
  sub_C53080(&qword_503FC20, "pipeliner-swap-branch-targets-mve", 33);
  LOBYTE(qword_503FCA8) = 0;
  qword_503FC50 = 59;
  LOBYTE(dword_503FC2C) = dword_503FC2C & 0x9F | 0x20;
  LOWORD(qword_503FCB8) = 256;
  qword_503FC48 = (__int64)"Swap target blocks of a conditional branch for MVE expander";
  sub_C53130(&qword_503FC20);
  return __cxa_atexit(sub_984900, &qword_503FC20, &qword_4A427C0);
}

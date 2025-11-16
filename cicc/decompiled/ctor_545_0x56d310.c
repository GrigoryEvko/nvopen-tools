// Function: ctor_545
// Address: 0x56d310
//
int ctor_545()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_501CE80 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_501CEFC = 1;
  qword_501CED0 = 0x100000000LL;
  dword_501CE8C &= 0x8000u;
  qword_501CE98 = 0;
  qword_501CEA0 = 0;
  qword_501CEA8 = 0;
  dword_501CE88 = v0;
  word_501CE90 = 0;
  qword_501CEB0 = 0;
  qword_501CEB8 = 0;
  qword_501CEC0 = 0;
  qword_501CEC8 = (__int64)&unk_501CED8;
  qword_501CEE0 = 0;
  qword_501CEE8 = (__int64)&unk_501CF00;
  qword_501CEF0 = 1;
  dword_501CEF8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_501CED0;
  v3 = (unsigned int)qword_501CED0 + 1LL;
  if ( v3 > HIDWORD(qword_501CED0) )
  {
    sub_C8D5F0((char *)&unk_501CED8 - 16, &unk_501CED8, v3, 8);
    v2 = (unsigned int)qword_501CED0;
  }
  *(_QWORD *)(qword_501CEC8 + 8 * v2) = v1;
  LODWORD(qword_501CED0) = qword_501CED0 + 1;
  qword_501CF08 = 0;
  qword_501CF10 = (__int64)&unk_49D9748;
  qword_501CF18 = 0;
  qword_501CE80 = (__int64)&unk_49DC090;
  qword_501CF20 = (__int64)&unk_49DC1D0;
  qword_501CF40 = (__int64)nullsub_23;
  qword_501CF38 = (__int64)sub_984030;
  sub_C53080(&qword_501CE80, "enable-complex-deinterleaving", 29);
  qword_501CEB0 = 41;
  qword_501CEA8 = (__int64)"Enable generation of complex instructions";
  LOWORD(qword_501CF18) = 257;
  LOBYTE(qword_501CF08) = 1;
  LOBYTE(dword_501CE8C) = dword_501CE8C & 0x9F | 0x20;
  sub_C53130(&qword_501CE80);
  return __cxa_atexit(sub_984900, &qword_501CE80, &qword_4A427C0);
}

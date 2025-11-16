// Function: ctor_436
// Address: 0x53bfe0
//
int ctor_436()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4FF8C40 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FF8CBC = 1;
  qword_4FF8C90 = 0x100000000LL;
  dword_4FF8C4C &= 0x8000u;
  qword_4FF8C58 = 0;
  qword_4FF8C60 = 0;
  qword_4FF8C68 = 0;
  dword_4FF8C48 = v0;
  word_4FF8C50 = 0;
  qword_4FF8C70 = 0;
  qword_4FF8C78 = 0;
  qword_4FF8C80 = 0;
  qword_4FF8C88 = (__int64)&unk_4FF8C98;
  qword_4FF8CA0 = 0;
  qword_4FF8CA8 = (__int64)&unk_4FF8CC0;
  qword_4FF8CB0 = 1;
  dword_4FF8CB8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FF8C90;
  v3 = (unsigned int)qword_4FF8C90 + 1LL;
  if ( v3 > HIDWORD(qword_4FF8C90) )
  {
    sub_C8D5F0((char *)&unk_4FF8C98 - 16, &unk_4FF8C98, v3, 8);
    v2 = (unsigned int)qword_4FF8C90;
  }
  *(_QWORD *)(qword_4FF8C88 + 8 * v2) = v1;
  LODWORD(qword_4FF8C90) = qword_4FF8C90 + 1;
  qword_4FF8CC8 = 0;
  qword_4FF8CD0 = (__int64)&unk_49D9748;
  qword_4FF8CD8 = 0;
  qword_4FF8C40 = (__int64)&unk_49DC090;
  qword_4FF8CE0 = (__int64)&unk_49DC1D0;
  qword_4FF8D00 = (__int64)nullsub_23;
  qword_4FF8CF8 = (__int64)sub_984030;
  sub_C53080(&qword_4FF8C40, "strip-global-constants", 22);
  LOBYTE(qword_4FF8CC8) = 0;
  LOWORD(qword_4FF8CD8) = 256;
  qword_4FF8C70 = 76;
  LOBYTE(dword_4FF8C4C) = dword_4FF8C4C & 0x9F | 0x20;
  qword_4FF8C68 = (__int64)"Removes debug compile units which reference to non-existing global constants";
  sub_C53130(&qword_4FF8C40);
  return __cxa_atexit(sub_984900, &qword_4FF8C40, &qword_4A427C0);
}

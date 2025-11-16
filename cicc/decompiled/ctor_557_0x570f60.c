// Function: ctor_557
// Address: 0x570f60
//
int ctor_557()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_501E920 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_501E99C = 1;
  qword_501E970 = 0x100000000LL;
  dword_501E92C &= 0x8000u;
  qword_501E938 = 0;
  qword_501E940 = 0;
  qword_501E948 = 0;
  dword_501E928 = v0;
  word_501E930 = 0;
  qword_501E950 = 0;
  qword_501E958 = 0;
  qword_501E960 = 0;
  qword_501E968 = (__int64)&unk_501E978;
  qword_501E980 = 0;
  qword_501E988 = (__int64)&unk_501E9A0;
  qword_501E990 = 1;
  dword_501E998 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_501E970;
  v3 = (unsigned int)qword_501E970 + 1LL;
  if ( v3 > HIDWORD(qword_501E970) )
  {
    sub_C8D5F0((char *)&unk_501E978 - 16, &unk_501E978, v3, 8);
    v2 = (unsigned int)qword_501E970;
  }
  *(_QWORD *)(qword_501E968 + 8 * v2) = v1;
  LODWORD(qword_501E970) = qword_501E970 + 1;
  qword_501E9A8 = 0;
  qword_501E9B0 = (__int64)&unk_49D9748;
  qword_501E9B8 = 0;
  qword_501E920 = (__int64)&unk_49DC090;
  qword_501E9C0 = (__int64)&unk_49DC1D0;
  qword_501E9E0 = (__int64)nullsub_23;
  qword_501E9D8 = (__int64)sub_984030;
  sub_C53080(&qword_501E920, "live-debug-variables", 20);
  LOBYTE(qword_501E9A8) = 1;
  LOWORD(qword_501E9B8) = 257;
  qword_501E948 = (__int64)"Enable the live debug variables pass";
  qword_501E950 = 36;
  LOBYTE(dword_501E92C) = dword_501E92C & 0x9F | 0x20;
  sub_C53130(&qword_501E920);
  return __cxa_atexit(sub_984900, &qword_501E920, &qword_4A427C0);
}

// Function: ctor_523
// Address: 0x563320
//
int ctor_523()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_50110A0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50110AC &= 0x8000u;
  word_50110B0 = 0;
  qword_50110F0 = 0x100000000LL;
  qword_50110B8 = 0;
  qword_50110C0 = 0;
  qword_50110C8 = 0;
  dword_50110A8 = v0;
  qword_50110D0 = 0;
  qword_50110D8 = 0;
  qword_50110E0 = 0;
  qword_50110E8 = (__int64)&unk_50110F8;
  qword_5011100 = 0;
  qword_5011108 = (__int64)&unk_5011120;
  qword_5011110 = 1;
  dword_5011118 = 0;
  byte_501111C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_50110F0;
  v3 = (unsigned int)qword_50110F0 + 1LL;
  if ( v3 > HIDWORD(qword_50110F0) )
  {
    sub_C8D5F0((char *)&unk_50110F8 - 16, &unk_50110F8, v3, 8);
    v2 = (unsigned int)qword_50110F0;
  }
  *(_QWORD *)(qword_50110E8 + 8 * v2) = v1;
  LODWORD(qword_50110F0) = qword_50110F0 + 1;
  qword_5011128 = 0;
  qword_5011130 = (__int64)&unk_49D9748;
  qword_5011138 = 0;
  qword_50110A0 = (__int64)&unk_49DC090;
  qword_5011140 = (__int64)&unk_49DC1D0;
  qword_5011160 = (__int64)nullsub_23;
  qword_5011158 = (__int64)sub_984030;
  sub_C53080(&qword_50110A0, "nvvm-verify-show-info", 21);
  qword_50110D0 = 46;
  LOBYTE(dword_50110AC) = dword_50110AC & 0xF8 | 1;
  qword_50110C8 = (__int64)"Enable info messages in NVVM verification pass";
  sub_C53130(&qword_50110A0);
  return __cxa_atexit(sub_984900, &qword_50110A0, &qword_4A427C0);
}

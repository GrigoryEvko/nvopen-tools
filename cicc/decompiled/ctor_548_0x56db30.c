// Function: ctor_548
// Address: 0x56db30
//
int ctor_548()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_501D220 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_501D22C &= 0x8000u;
  word_501D230 = 0;
  qword_501D270 = 0x100000000LL;
  qword_501D238 = 0;
  qword_501D240 = 0;
  qword_501D248 = 0;
  dword_501D228 = v0;
  qword_501D250 = 0;
  qword_501D258 = 0;
  qword_501D260 = 0;
  qword_501D268 = (__int64)&unk_501D278;
  qword_501D280 = 0;
  qword_501D288 = (__int64)&unk_501D2A0;
  qword_501D290 = 1;
  dword_501D298 = 0;
  byte_501D29C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_501D270;
  v3 = (unsigned int)qword_501D270 + 1LL;
  if ( v3 > HIDWORD(qword_501D270) )
  {
    sub_C8D5F0((char *)&unk_501D278 - 16, &unk_501D278, v3, 8);
    v2 = (unsigned int)qword_501D270;
  }
  *(_QWORD *)(qword_501D268 + 8 * v2) = v1;
  LODWORD(qword_501D270) = qword_501D270 + 1;
  qword_501D2A8 = 0;
  qword_501D2B0 = (__int64)&unk_49D9728;
  qword_501D2B8 = 0;
  qword_501D220 = (__int64)&unk_49DBF10;
  qword_501D2C0 = (__int64)&unk_49DC290;
  qword_501D2E0 = (__int64)nullsub_24;
  qword_501D2D8 = (__int64)sub_984050;
  sub_C53080(&qword_501D220, "expand-div-rem-bits", 19);
  LODWORD(qword_501D2A8) = 0x800000;
  BYTE4(qword_501D2B8) = 1;
  LODWORD(qword_501D2B8) = 0x800000;
  qword_501D250 = 74;
  LOBYTE(dword_501D22C) = dword_501D22C & 0x9F | 0x20;
  qword_501D248 = (__int64)"div and rem instructions on integers with more than <N> bits are expanded.";
  sub_C53130(&qword_501D220);
  return __cxa_atexit(sub_984970, &qword_501D220, &qword_4A427C0);
}

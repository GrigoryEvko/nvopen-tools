// Function: ctor_491
// Address: 0x5543a0
//
int ctor_491()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_5007A40 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_5007A4C &= 0x8000u;
  word_5007A50 = 0;
  qword_5007A90 = 0x100000000LL;
  qword_5007A58 = 0;
  qword_5007A60 = 0;
  qword_5007A68 = 0;
  dword_5007A48 = v0;
  qword_5007A70 = 0;
  qword_5007A78 = 0;
  qword_5007A80 = 0;
  qword_5007A88 = (__int64)&unk_5007A98;
  qword_5007AA0 = 0;
  qword_5007AA8 = (__int64)&unk_5007AC0;
  qword_5007AB0 = 1;
  dword_5007AB8 = 0;
  byte_5007ABC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5007A90;
  v3 = (unsigned int)qword_5007A90 + 1LL;
  if ( v3 > HIDWORD(qword_5007A90) )
  {
    sub_C8D5F0((char *)&unk_5007A98 - 16, &unk_5007A98, v3, 8);
    v2 = (unsigned int)qword_5007A90;
  }
  *(_QWORD *)(qword_5007A88 + 8 * v2) = v1;
  LODWORD(qword_5007A90) = qword_5007A90 + 1;
  qword_5007AC8 = 0;
  qword_5007AD0 = (__int64)&unk_49D9748;
  qword_5007AD8 = 0;
  qword_5007A40 = (__int64)&unk_49DC090;
  qword_5007AE0 = (__int64)&unk_49DC1D0;
  qword_5007B00 = (__int64)nullsub_23;
  qword_5007AF8 = (__int64)sub_984030;
  sub_C53080(&qword_5007A40, "aggregate-extracted-args", 24);
  qword_5007A70 = 47;
  LOBYTE(dword_5007A4C) = dword_5007A4C & 0x9F | 0x20;
  qword_5007A68 = (__int64)"Aggregate arguments to code-extracted functions";
  sub_C53130(&qword_5007A40);
  return __cxa_atexit(sub_984900, &qword_5007A40, &qword_4A427C0);
}

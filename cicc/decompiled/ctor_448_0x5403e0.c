// Function: ctor_448
// Address: 0x5403e0
//
int ctor_448()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4FFB2C0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FFB2CC &= 0x8000u;
  word_4FFB2D0 = 0;
  qword_4FFB310 = 0x100000000LL;
  qword_4FFB2D8 = 0;
  qword_4FFB2E0 = 0;
  qword_4FFB2E8 = 0;
  dword_4FFB2C8 = v0;
  qword_4FFB2F0 = 0;
  qword_4FFB2F8 = 0;
  qword_4FFB300 = 0;
  qword_4FFB308 = (__int64)&unk_4FFB318;
  qword_4FFB320 = 0;
  qword_4FFB328 = (__int64)&unk_4FFB340;
  qword_4FFB330 = 1;
  dword_4FFB338 = 0;
  byte_4FFB33C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FFB310;
  v3 = (unsigned int)qword_4FFB310 + 1LL;
  if ( v3 > HIDWORD(qword_4FFB310) )
  {
    sub_C8D5F0((char *)&unk_4FFB318 - 16, &unk_4FFB318, v3, 8);
    v2 = (unsigned int)qword_4FFB310;
  }
  *(_QWORD *)(qword_4FFB308 + 8 * v2) = v1;
  LODWORD(qword_4FFB310) = qword_4FFB310 + 1;
  qword_4FFB348 = 0;
  qword_4FFB350 = (__int64)&unk_49D9728;
  qword_4FFB358 = 0;
  qword_4FFB2C0 = (__int64)&unk_49DBF10;
  qword_4FFB360 = (__int64)&unk_49DC290;
  qword_4FFB380 = (__int64)nullsub_24;
  qword_4FFB378 = (__int64)sub_984050;
  sub_C53080(&qword_4FFB2C0, "float2int-max-integer-bw", 24);
  LODWORD(qword_4FFB348) = 64;
  BYTE4(qword_4FFB358) = 1;
  LODWORD(qword_4FFB358) = 64;
  qword_4FFB2F0 = 57;
  LOBYTE(dword_4FFB2CC) = dword_4FFB2CC & 0x9F | 0x20;
  qword_4FFB2E8 = (__int64)"Max integer bitwidth to consider in float2int(default=64)";
  sub_C53130(&qword_4FFB2C0);
  return __cxa_atexit(sub_984970, &qword_4FFB2C0, &qword_4A427C0);
}

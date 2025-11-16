// Function: ctor_511
// Address: 0x55d3f0
//
int ctor_511()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_500C000 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500C018 = 0;
  qword_500C020 = 0;
  qword_500C028 = 0;
  qword_500C030 = 0;
  dword_500C00C = dword_500C00C & 0x8000 | 1;
  word_500C010 = 0;
  qword_500C050 = 0x100000000LL;
  dword_500C008 = v0;
  qword_500C038 = 0;
  qword_500C040 = 0;
  qword_500C048 = (__int64)&unk_500C058;
  qword_500C060 = 0;
  qword_500C068 = (__int64)&unk_500C080;
  qword_500C070 = 1;
  dword_500C078 = 0;
  byte_500C07C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_500C050;
  v3 = (unsigned int)qword_500C050 + 1LL;
  if ( v3 > HIDWORD(qword_500C050) )
  {
    sub_C8D5F0((char *)&unk_500C058 - 16, &unk_500C058, v3, 8);
    v2 = (unsigned int)qword_500C050;
  }
  *(_QWORD *)(qword_500C048 + 8 * v2) = v1;
  LODWORD(qword_500C050) = qword_500C050 + 1;
  qword_500C088 = 0;
  qword_500C000 = (__int64)&unk_49DAD08;
  qword_500C090 = 0;
  qword_500C098 = 0;
  qword_500C0D8 = (__int64)&unk_49DC350;
  qword_500C0A0 = 0;
  qword_500C0F8 = (__int64)nullsub_81;
  qword_500C0A8 = 0;
  qword_500C0F0 = (__int64)sub_BB8600;
  qword_500C0B0 = 0;
  byte_500C0B8 = 0;
  qword_500C0C0 = 0;
  qword_500C0C8 = 0;
  qword_500C0D0 = 0;
  sub_C53080(&qword_500C000, "rewrite-map-file", 16);
  qword_500C030 = 18;
  qword_500C028 = (__int64)"Symbol Rewrite Map";
  qword_500C038 = (__int64)"filename";
  qword_500C040 = 8;
  LOBYTE(dword_500C00C) = dword_500C00C & 0x9F | 0x20;
  sub_C53130(&qword_500C000);
  return __cxa_atexit(sub_BB89D0, &qword_500C000, &qword_4A427C0);
}

// Function: ctor_567
// Address: 0x573680
//
int ctor_567()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_5020020 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_502009C = 1;
  qword_5020070 = 0x100000000LL;
  dword_502002C &= 0x8000u;
  qword_5020038 = 0;
  qword_5020040 = 0;
  qword_5020048 = 0;
  dword_5020028 = v0;
  word_5020030 = 0;
  qword_5020050 = 0;
  qword_5020058 = 0;
  qword_5020060 = 0;
  qword_5020068 = (__int64)&unk_5020078;
  qword_5020080 = 0;
  qword_5020088 = (__int64)&unk_50200A0;
  qword_5020090 = 1;
  dword_5020098 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5020070;
  v3 = (unsigned int)qword_5020070 + 1LL;
  if ( v3 > HIDWORD(qword_5020070) )
  {
    sub_C8D5F0((char *)&unk_5020078 - 16, &unk_5020078, v3, 8);
    v2 = (unsigned int)qword_5020070;
  }
  *(_QWORD *)(qword_5020068 + 8 * v2) = v1;
  LODWORD(qword_5020070) = qword_5020070 + 1;
  qword_50200A8 = 0;
  qword_50200B0 = (__int64)&unk_49D9748;
  qword_50200B8 = 0;
  qword_5020020 = (__int64)&unk_49DC090;
  qword_50200C0 = (__int64)&unk_49DC1D0;
  qword_50200E0 = (__int64)nullsub_23;
  qword_50200D8 = (__int64)sub_984030;
  sub_C53080(&qword_5020020, "dropped-variable-stats-mir", 26);
  qword_5020050 = 49;
  LOBYTE(qword_50200A8) = 0;
  LOBYTE(dword_502002C) = dword_502002C & 0x9F | 0x20;
  qword_5020048 = (__int64)"Dump dropped debug variables stats for MIR passes";
  LOWORD(qword_50200B8) = 256;
  sub_C53130(&qword_5020020);
  return __cxa_atexit(sub_984900, &qword_5020020, &qword_4A427C0);
}

// Function: ctor_506
// Address: 0x55b920
//
int ctor_506()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_500B040 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_500B04C &= 0x8000u;
  word_500B050 = 0;
  qword_500B090 = 0x100000000LL;
  qword_500B058 = 0;
  qword_500B060 = 0;
  qword_500B068 = 0;
  dword_500B048 = v0;
  qword_500B070 = 0;
  qword_500B078 = 0;
  qword_500B080 = 0;
  qword_500B088 = (__int64)&unk_500B098;
  qword_500B0A0 = 0;
  qword_500B0A8 = (__int64)&unk_500B0C0;
  qword_500B0B0 = 1;
  dword_500B0B8 = 0;
  byte_500B0BC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_500B090;
  v3 = (unsigned int)qword_500B090 + 1LL;
  if ( v3 > HIDWORD(qword_500B090) )
  {
    sub_C8D5F0((char *)&unk_500B098 - 16, &unk_500B098, v3, 8);
    v2 = (unsigned int)qword_500B090;
  }
  *(_QWORD *)(qword_500B088 + 8 * v2) = v1;
  LODWORD(qword_500B090) = qword_500B090 + 1;
  qword_500B0C8 = 0;
  qword_500B0D0 = (__int64)&unk_49D9728;
  qword_500B0D8 = 0;
  qword_500B040 = (__int64)&unk_49DBF10;
  qword_500B0E0 = (__int64)&unk_49DC290;
  qword_500B100 = (__int64)nullsub_24;
  qword_500B0F8 = (__int64)sub_984050;
  sub_C53080(&qword_500B040, "move-auto-init-threshold", 24);
  LODWORD(qword_500B0C8) = 128;
  BYTE4(qword_500B0D8) = 1;
  LODWORD(qword_500B0D8) = 128;
  qword_500B070 = 56;
  LOBYTE(dword_500B04C) = dword_500B04C & 0x9F | 0x20;
  qword_500B068 = (__int64)"Maximum instructions to analyze per moved initialization";
  sub_C53130(&qword_500B040);
  return __cxa_atexit(sub_984970, &qword_500B040, &qword_4A427C0);
}

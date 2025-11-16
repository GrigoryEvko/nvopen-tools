// Function: ctor_344
// Address: 0x50b2a0
//
int ctor_344()
{
  int v0; // edx

  qword_4FCEFA0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCEFAC &= 0xF000u;
  qword_4FCEFB0 = 0;
  qword_4FCEFE8 = (__int64)qword_4FA01C0;
  qword_4FCEFB8 = 0;
  qword_4FCEFC0 = 0;
  qword_4FCEFC8 = 0;
  dword_4FCEFA8 = v0;
  qword_4FCEFF8 = (__int64)&unk_4FCF018;
  qword_4FCF000 = (__int64)&unk_4FCF018;
  qword_4FCEFD0 = 0;
  qword_4FCEFD8 = 0;
  qword_4FCF048 = (__int64)&unk_49E74E8;
  word_4FCF050 = 256;
  qword_4FCEFE0 = 0;
  qword_4FCEFF0 = 0;
  qword_4FCEFA0 = (__int64)&unk_49EEC70;
  qword_4FCF008 = 4;
  byte_4FCF038 = 0;
  qword_4FCF058 = (__int64)&unk_49EEDB0;
  dword_4FCF010 = 0;
  byte_4FCF040 = 0;
  sub_16B8280(&qword_4FCEFA0, "dag-dump-verbose", 16);
  qword_4FCEFD0 = 58;
  LOBYTE(word_4FCEFAC) = word_4FCEFAC & 0x9F | 0x20;
  qword_4FCEFC8 = (__int64)"Display more information when dumping selection DAG nodes.";
  sub_16B88A0(&qword_4FCEFA0);
  return __cxa_atexit(sub_12EDEC0, &qword_4FCEFA0, &qword_4A427C0);
}

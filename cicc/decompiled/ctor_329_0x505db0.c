// Function: ctor_329
// Address: 0x505db0
//
int ctor_329()
{
  int v0; // edx

  qword_4FCAAE0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCAAEC &= 0xF000u;
  qword_4FCAAF0 = 0;
  qword_4FCAB28 = (__int64)qword_4FA01C0;
  qword_4FCAAF8 = 0;
  qword_4FCAB00 = 0;
  qword_4FCAB08 = 0;
  dword_4FCAAE8 = v0;
  qword_4FCAB38 = (__int64)&unk_4FCAB58;
  qword_4FCAB40 = (__int64)&unk_4FCAB58;
  qword_4FCAB10 = 0;
  qword_4FCAB18 = 0;
  qword_4FCAB88 = (__int64)&unk_49E74E8;
  word_4FCAB90 = 256;
  qword_4FCAB20 = 0;
  qword_4FCAB30 = 0;
  qword_4FCAAE0 = (__int64)&unk_49EEC70;
  qword_4FCAB48 = 4;
  byte_4FCAB78 = 0;
  qword_4FCAB98 = (__int64)&unk_49EEDB0;
  dword_4FCAB50 = 0;
  byte_4FCAB80 = 0;
  sub_16B8280(&qword_4FCAAE0, "enable-patchpoint-liveness", 26);
  word_4FCAB90 = 257;
  byte_4FCAB80 = 1;
  qword_4FCAB10 = 40;
  LOBYTE(word_4FCAAEC) = word_4FCAAEC & 0x9F | 0x20;
  qword_4FCAB08 = (__int64)"Enable PatchPoint Liveness Analysis Pass";
  sub_16B88A0(&qword_4FCAAE0);
  return __cxa_atexit(sub_12EDEC0, &qword_4FCAAE0, &qword_4A427C0);
}

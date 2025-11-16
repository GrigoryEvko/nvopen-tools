// Function: ctor_234
// Address: 0x4ebb70
//
int ctor_234()
{
  int v0; // edx

  qword_4FB6280 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB628C &= 0xF000u;
  qword_4FB6290 = 0;
  qword_4FB62C8 = (__int64)qword_4FA01C0;
  qword_4FB6298 = 0;
  qword_4FB62A0 = 0;
  qword_4FB62A8 = 0;
  dword_4FB6288 = v0;
  qword_4FB62D8 = (__int64)&unk_4FB62F8;
  qword_4FB62E0 = (__int64)&unk_4FB62F8;
  qword_4FB62B0 = 0;
  qword_4FB62B8 = 0;
  qword_4FB6328 = (__int64)&unk_49E74E8;
  word_4FB6330 = 256;
  qword_4FB62C0 = 0;
  qword_4FB62D0 = 0;
  qword_4FB6280 = (__int64)&unk_49EEC70;
  qword_4FB62E8 = 4;
  byte_4FB6318 = 0;
  qword_4FB6338 = (__int64)&unk_49EEDB0;
  dword_4FB62F0 = 0;
  byte_4FB6320 = 0;
  sub_16B8280(&qword_4FB6280, "aggregate-extracted-args", 24);
  qword_4FB62B0 = 47;
  LOBYTE(word_4FB628C) = word_4FB628C & 0x9F | 0x20;
  qword_4FB62A8 = (__int64)"Aggregate arguments to code-extracted functions";
  sub_16B88A0(&qword_4FB6280);
  return __cxa_atexit(sub_12EDEC0, &qword_4FB6280, &qword_4A427C0);
}

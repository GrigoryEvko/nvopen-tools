// Function: ctor_294
// Address: 0x4fcb40
//
int ctor_294()
{
  int v0; // edx

  qword_4FC4360 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC436C &= 0xF000u;
  qword_4FC4370 = 0;
  qword_4FC43A8 = (__int64)qword_4FA01C0;
  qword_4FC4378 = 0;
  qword_4FC4380 = 0;
  qword_4FC4388 = 0;
  dword_4FC4368 = v0;
  qword_4FC43B8 = (__int64)&unk_4FC43D8;
  qword_4FC43C0 = (__int64)&unk_4FC43D8;
  qword_4FC4390 = 0;
  qword_4FC4398 = 0;
  qword_4FC4408 = (__int64)&unk_49E74E8;
  word_4FC4410 = 256;
  qword_4FC43A0 = 0;
  qword_4FC43B0 = 0;
  qword_4FC4360 = (__int64)&unk_49EEC70;
  qword_4FC43C8 = 4;
  byte_4FC43F8 = 0;
  qword_4FC4418 = (__int64)&unk_49EEDB0;
  dword_4FC43D0 = 0;
  byte_4FC4400 = 0;
  sub_16B8280(&qword_4FC4360, "live-debug-variables", 20);
  qword_4FC4388 = (__int64)"Enable the live debug variables pass";
  word_4FC4410 = 257;
  byte_4FC4400 = 1;
  qword_4FC4390 = 36;
  LOBYTE(word_4FC436C) = word_4FC436C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC4360);
  return __cxa_atexit(sub_12EDEC0, &qword_4FC4360, &qword_4A427C0);
}

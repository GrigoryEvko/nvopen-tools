// Function: ctor_159
// Address: 0x4cf840
//
int ctor_159()
{
  int v0; // edx

  qword_4FA0400 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA040C &= 0xF000u;
  qword_4FA0410 = 0;
  qword_4FA0448 = (__int64)qword_4FA01C0;
  qword_4FA0418 = 0;
  qword_4FA0420 = 0;
  qword_4FA0428 = 0;
  dword_4FA0408 = v0;
  qword_4FA0458 = (__int64)&unk_4FA0478;
  qword_4FA0460 = (__int64)&unk_4FA0478;
  qword_4FA0430 = 0;
  qword_4FA0438 = 0;
  qword_4FA04A8 = (__int64)&unk_49E74E8;
  word_4FA04B0 = 256;
  qword_4FA0440 = 0;
  qword_4FA0450 = 0;
  qword_4FA0400 = (__int64)&unk_49EEC70;
  qword_4FA0468 = 4;
  byte_4FA0498 = 0;
  qword_4FA04B8 = (__int64)&unk_49EEDB0;
  dword_4FA0470 = 0;
  byte_4FA04A0 = 0;
  sub_16B8280(&qword_4FA0400, "view-background", 15);
  qword_4FA0430 = 64;
  LOBYTE(word_4FA040C) = word_4FA040C & 0x9F | 0x20;
  qword_4FA0428 = (__int64)"Execute graph viewer in the background. Creates tmp file litter.";
  sub_16B88A0(&qword_4FA0400);
  return __cxa_atexit(sub_12EDEC0, &qword_4FA0400, &qword_4A427C0);
}

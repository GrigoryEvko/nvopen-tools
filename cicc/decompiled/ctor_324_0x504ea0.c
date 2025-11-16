// Function: ctor_324
// Address: 0x504ea0
//
int ctor_324()
{
  int v0; // edx

  qword_4FCA220 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCA22C &= 0xF000u;
  qword_4FCA230 = 0;
  qword_4FCA268 = (__int64)qword_4FA01C0;
  qword_4FCA238 = 0;
  qword_4FCA240 = 0;
  qword_4FCA248 = 0;
  dword_4FCA228 = v0;
  qword_4FCA278 = (__int64)&unk_4FCA298;
  qword_4FCA280 = (__int64)&unk_4FCA298;
  qword_4FCA250 = 0;
  qword_4FCA258 = 0;
  qword_4FCA2C8 = (__int64)&unk_49E74E8;
  word_4FCA2D0 = 256;
  qword_4FCA260 = 0;
  qword_4FCA270 = 0;
  qword_4FCA220 = (__int64)&unk_49EEC70;
  qword_4FCA288 = 4;
  byte_4FCA2B8 = 0;
  qword_4FCA2D8 = (__int64)&unk_49EEDB0;
  dword_4FCA290 = 0;
  byte_4FCA2C0 = 0;
  sub_16B8280(&qword_4FCA220, "safestack-use-pointer-address", 29);
  word_4FCA2D0 = 256;
  byte_4FCA2C0 = 0;
  LOBYTE(word_4FCA22C) = word_4FCA22C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FCA220);
  return __cxa_atexit(sub_12EDEC0, &qword_4FCA220, &qword_4A427C0);
}

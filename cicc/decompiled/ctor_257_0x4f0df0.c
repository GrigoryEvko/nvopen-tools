// Function: ctor_257
// Address: 0x4f0df0
//
int ctor_257()
{
  int v0; // edx

  qword_4FBA4A0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FBA4AC &= 0xF000u;
  qword_4FBA4B0 = 0;
  qword_4FBA4E8 = (__int64)qword_4FA01C0;
  qword_4FBA4B8 = 0;
  qword_4FBA4C0 = 0;
  qword_4FBA4C8 = 0;
  dword_4FBA4A8 = v0;
  qword_4FBA4F8 = (__int64)&unk_4FBA518;
  qword_4FBA500 = (__int64)&unk_4FBA518;
  qword_4FBA4D0 = 0;
  qword_4FBA4D8 = 0;
  qword_4FBA548 = (__int64)&unk_49E74E8;
  word_4FBA550 = 256;
  qword_4FBA4E0 = 0;
  qword_4FBA4F0 = 0;
  qword_4FBA4A0 = (__int64)&unk_49EEC70;
  qword_4FBA508 = 4;
  byte_4FBA538 = 0;
  qword_4FBA558 = (__int64)&unk_49EEDB0;
  dword_4FBA510 = 0;
  byte_4FBA540 = 0;
  sub_16B8280(&qword_4FBA4A0, "nvvm-verify-show-info", 21);
  qword_4FBA4D0 = 46;
  LOBYTE(word_4FBA4AC) = word_4FBA4AC & 0xF8 | 1;
  qword_4FBA4C8 = (__int64)"Enable info messages in NVVM verification pass";
  sub_16B88A0(&qword_4FBA4A0);
  return __cxa_atexit(sub_12EDEC0, &qword_4FBA4A0, &qword_4A427C0);
}

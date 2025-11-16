// Function: ctor_146
// Address: 0x4cc5c0
//
int ctor_146()
{
  int v0; // edx

  qword_4F9E160 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9E16C &= 0xF000u;
  qword_4F9E170 = 0;
  qword_4F9E1A8 = (__int64)&unk_4FA01C0;
  qword_4F9E178 = 0;
  qword_4F9E180 = 0;
  qword_4F9E188 = 0;
  dword_4F9E168 = v0;
  qword_4F9E1B8 = (__int64)&unk_4F9E1D8;
  qword_4F9E1C0 = (__int64)&unk_4F9E1D8;
  qword_4F9E190 = 0;
  qword_4F9E198 = 0;
  qword_4F9E208 = (__int64)&unk_49E74E8;
  word_4F9E210 = 256;
  qword_4F9E1A0 = 0;
  qword_4F9E1B0 = 0;
  qword_4F9E160 = (__int64)&unk_49EEC70;
  qword_4F9E1C8 = 4;
  byte_4F9E1F8 = 0;
  qword_4F9E218 = (__int64)&unk_49EEDB0;
  dword_4F9E1D0 = 0;
  byte_4F9E200 = 0;
  sub_16B8280(&qword_4F9E160, "disable-ipo-derefinement", 24);
  word_4F9E210 = 256;
  byte_4F9E200 = 0;
  qword_4F9E190 = 96;
  LOBYTE(word_4F9E16C) = word_4F9E16C & 0x9F | 0x20;
  qword_4F9E188 = (__int64)"Stop inter-procedural optimizations on linkonce_odr/weak_odr functions that may get derefinement";
  sub_16B88A0(&qword_4F9E160);
  return __cxa_atexit(sub_12EDEC0, &qword_4F9E160, &qword_4A427C0);
}

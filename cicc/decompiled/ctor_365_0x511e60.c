// Function: ctor_365
// Address: 0x511e60
//
int ctor_365()
{
  int v0; // edx

  qword_4FD4140 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FD414C &= 0xF000u;
  qword_4FD4150 = 0;
  qword_4FD4188 = (__int64)qword_4FA01C0;
  qword_4FD4158 = 0;
  qword_4FD4160 = 0;
  qword_4FD4168 = 0;
  dword_4FD4148 = v0;
  qword_4FD4198 = (__int64)&unk_4FD41B8;
  qword_4FD41A0 = (__int64)&unk_4FD41B8;
  qword_4FD4170 = 0;
  qword_4FD4178 = 0;
  qword_4FD41E8 = (__int64)&unk_49E74E8;
  word_4FD41F0 = 256;
  qword_4FD4180 = 0;
  qword_4FD4190 = 0;
  qword_4FD4140 = (__int64)&unk_49EEC70;
  qword_4FD41A8 = 4;
  byte_4FD41D8 = 0;
  qword_4FD41F8 = (__int64)&unk_49EEDB0;
  dword_4FD41B0 = 0;
  byte_4FD41E0 = 0;
  sub_16B8280(&qword_4FD4140, "cta-reconfig-aware-mrpa", 23);
  word_4FD41F0 = 257;
  byte_4FD41E0 = 1;
  qword_4FD4170 = 61;
  LOBYTE(word_4FD414C) = word_4FD414C & 0x9F | 0x20;
  qword_4FD4168 = (__int64)"Enable CTA reconfig aware machine register pressure analysis.";
  sub_16B88A0(&qword_4FD4140);
  return __cxa_atexit(sub_12EDEC0, &qword_4FD4140, &qword_4A427C0);
}

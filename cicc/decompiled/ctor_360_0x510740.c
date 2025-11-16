// Function: ctor_360
// Address: 0x510740
//
int ctor_360()
{
  int v0; // edx

  qword_4FD2A60 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FD2A6C &= 0xF000u;
  qword_4FD2A70 = 0;
  qword_4FD2AA8 = (__int64)qword_4FA01C0;
  qword_4FD2A78 = 0;
  qword_4FD2A80 = 0;
  qword_4FD2A88 = 0;
  dword_4FD2A68 = v0;
  qword_4FD2AB8 = (__int64)&unk_4FD2AD8;
  qword_4FD2AC0 = (__int64)&unk_4FD2AD8;
  qword_4FD2A90 = 0;
  qword_4FD2A98 = 0;
  qword_4FD2B08 = (__int64)&unk_49E74E8;
  word_4FD2B10 = 256;
  qword_4FD2AA0 = 0;
  qword_4FD2AB0 = 0;
  qword_4FD2A60 = (__int64)&unk_49EEC70;
  qword_4FD2AC8 = 4;
  byte_4FD2AF8 = 0;
  qword_4FD2B18 = (__int64)&unk_49EEDB0;
  dword_4FD2AD0 = 0;
  byte_4FD2B00 = 0;
  sub_16B8280(&qword_4FD2A60, "enable-bfi64", 12);
  word_4FD2B10 = 257;
  byte_4FD2B00 = 1;
  qword_4FD2A90 = 44;
  LOBYTE(word_4FD2A6C) = word_4FD2A6C & 0x9F | 0x20;
  qword_4FD2A88 = (__int64)"Enable generation of 64-bit BFI instructions";
  sub_16B88A0(&qword_4FD2A60);
  return __cxa_atexit(sub_12EDEC0, &qword_4FD2A60, &qword_4A427C0);
}

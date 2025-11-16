// Function: ctor_138
// Address: 0x4b3430
//
int ctor_138()
{
  int v0; // edx

  qword_4F9D780 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9D78C &= 0xF000u;
  qword_4F9D790 = 0;
  qword_4F9D7C8 = (__int64)&unk_4FA01C0;
  qword_4F9D798 = 0;
  qword_4F9D7A0 = 0;
  qword_4F9D7A8 = 0;
  dword_4F9D788 = v0;
  qword_4F9D7D8 = (__int64)&unk_4F9D7F8;
  qword_4F9D7E0 = (__int64)&unk_4F9D7F8;
  qword_4F9D7B0 = 0;
  qword_4F9D7B8 = 0;
  qword_4F9D828 = (__int64)&unk_49E74E8;
  word_4F9D830 = 256;
  qword_4F9D7C0 = 0;
  qword_4F9D7D0 = 0;
  qword_4F9D780 = (__int64)&unk_49EEC70;
  qword_4F9D7E8 = 4;
  byte_4F9D818 = 0;
  qword_4F9D838 = (__int64)&unk_49EEDB0;
  dword_4F9D7F0 = 0;
  byte_4F9D820 = 0;
  sub_16B8280(&qword_4F9D780, "verify-assumption-cache", 23);
  word_4F9D830 = 256;
  byte_4F9D820 = 0;
  qword_4F9D7B0 = 39;
  LOBYTE(word_4F9D78C) = word_4F9D78C & 0x9F | 0x20;
  qword_4F9D7A8 = (__int64)"Enable verification of assumption cache";
  sub_16B88A0(&qword_4F9D780);
  return __cxa_atexit(sub_12EDEC0, &qword_4F9D780, &qword_4A427C0);
}

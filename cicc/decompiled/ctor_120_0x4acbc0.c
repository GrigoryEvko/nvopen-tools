// Function: ctor_120
// Address: 0x4acbc0
//
int ctor_120()
{
  int v0; // edx

  qword_4F98E80 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F98E8C &= 0xF000u;
  qword_4F98E90 = 0;
  qword_4F98EC8 = (__int64)&unk_4FA01C0;
  qword_4F98E98 = 0;
  qword_4F98EA0 = 0;
  qword_4F98EA8 = 0;
  dword_4F98E88 = v0;
  qword_4F98ED8 = (__int64)&unk_4F98EF8;
  qword_4F98EE0 = (__int64)&unk_4F98EF8;
  qword_4F98EB0 = 0;
  qword_4F98EB8 = 0;
  qword_4F98F28 = (__int64)&unk_49E74E8;
  word_4F98F30 = 256;
  qword_4F98EC0 = 0;
  qword_4F98ED0 = 0;
  qword_4F98E80 = (__int64)&unk_49EEC70;
  qword_4F98EE8 = 4;
  byte_4F98F18 = 0;
  qword_4F98F38 = (__int64)&unk_49EEDB0;
  dword_4F98EF0 = 0;
  byte_4F98F20 = 0;
  sub_16B8280(&qword_4F98E80, "enable-unsafe-globalsmodref-alias-results", 41);
  word_4F98F30 = 256;
  byte_4F98F20 = 0;
  LOBYTE(word_4F98E8C) = word_4F98E8C & 0x9F | 0x20;
  sub_16B88A0(&qword_4F98E80);
  return __cxa_atexit(sub_12EDEC0, &qword_4F98E80, &qword_4A427C0);
}

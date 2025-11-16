// Function: ctor_341
// Address: 0x50aa80
//
int ctor_341()
{
  int v0; // edx

  qword_4FCEB40 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCEB4C &= 0xF000u;
  qword_4FCEB50 = 0;
  qword_4FCEB88 = (__int64)qword_4FA01C0;
  qword_4FCEB58 = 0;
  qword_4FCEB60 = 0;
  qword_4FCEB68 = 0;
  dword_4FCEB48 = v0;
  qword_4FCEB98 = (__int64)&unk_4FCEBB8;
  qword_4FCEBA0 = (__int64)&unk_4FCEBB8;
  qword_4FCEB70 = 0;
  qword_4FCEB78 = 0;
  qword_4FCEBE8 = (__int64)&unk_49E74E8;
  word_4FCEBF0 = 256;
  qword_4FCEB80 = 0;
  qword_4FCEB90 = 0;
  qword_4FCEB40 = (__int64)&unk_49EEC70;
  qword_4FCEBA8 = 4;
  byte_4FCEBD8 = 0;
  qword_4FCEBF8 = (__int64)&unk_49EEDB0;
  dword_4FCEBB0 = 0;
  byte_4FCEBE0 = 0;
  sub_16B8280(&qword_4FCEB40, "enable-legalize-types-checking", 30);
  LOBYTE(word_4FCEB4C) = word_4FCEB4C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FCEB40);
  return __cxa_atexit(sub_12EDEC0, &qword_4FCEB40, &qword_4A427C0);
}

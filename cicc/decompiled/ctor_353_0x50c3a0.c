// Function: ctor_353
// Address: 0x50c3a0
//
int ctor_353()
{
  int v0; // edx

  qword_4FCFA60 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCFA6C &= 0xF000u;
  qword_4FCFA70 = 0;
  qword_4FCFAA8 = (__int64)qword_4FA01C0;
  qword_4FCFA78 = 0;
  qword_4FCFA80 = 0;
  qword_4FCFA88 = 0;
  dword_4FCFA68 = v0;
  qword_4FCFAB8 = (__int64)&unk_4FCFAD8;
  qword_4FCFAC0 = (__int64)&unk_4FCFAD8;
  qword_4FCFA90 = 0;
  qword_4FCFA98 = 0;
  qword_4FCFB08 = (__int64)&unk_49E74E8;
  word_4FCFB10 = 256;
  qword_4FCFAA0 = 0;
  qword_4FCFAB0 = 0;
  qword_4FCFA60 = (__int64)&unk_49EEC70;
  qword_4FCFAC8 = 4;
  byte_4FCFAF8 = 0;
  qword_4FCFB18 = (__int64)&unk_49EEDB0;
  dword_4FCFAD0 = 0;
  byte_4FCFB00 = 0;
  sub_16B8280(&qword_4FCFA60, "print-regusage", 14);
  word_4FCFB10 = 256;
  byte_4FCFB00 = 0;
  qword_4FCFA90 = 52;
  LOBYTE(word_4FCFA6C) = word_4FCFA6C & 0x9F | 0x20;
  qword_4FCFA88 = (__int64)"print register usage details collected for analysis.";
  sub_16B88A0(&qword_4FCFA60);
  return __cxa_atexit(sub_12EDEC0, &qword_4FCFA60, &qword_4A427C0);
}

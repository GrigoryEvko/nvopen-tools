// Function: ctor_342
// Address: 0x50abf0
//
int ctor_342()
{
  int v0; // eax
  int v1; // eax

  qword_4FCED00 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCED0C &= 0xF000u;
  qword_4FCED48 = (__int64)qword_4FA01C0;
  qword_4FCED10 = 0;
  qword_4FCED18 = 0;
  qword_4FCED20 = 0;
  dword_4FCED08 = v0;
  qword_4FCED58 = (__int64)&unk_4FCED78;
  qword_4FCED60 = (__int64)&unk_4FCED78;
  qword_4FCED28 = 0;
  qword_4FCED30 = 0;
  qword_4FCEDA8 = (__int64)&unk_49E74E8;
  word_4FCEDB0 = 256;
  qword_4FCED38 = 0;
  qword_4FCED40 = 0;
  qword_4FCED00 = (__int64)&unk_49EEC70;
  qword_4FCED50 = 0;
  byte_4FCED98 = 0;
  qword_4FCEDB8 = (__int64)&unk_49EEDB0;
  qword_4FCED68 = 4;
  dword_4FCED70 = 0;
  byte_4FCEDA0 = 0;
  sub_16B8280(&qword_4FCED00, "disable-dfa-sched", 17);
  word_4FCEDB0 = 256;
  byte_4FCEDA0 = 0;
  qword_4FCED30 = 36;
  LOBYTE(word_4FCED0C) = word_4FCED0C & 0x98 | 0x21;
  qword_4FCED28 = (__int64)"Disable use of DFA during scheduling";
  sub_16B88A0(&qword_4FCED00);
  __cxa_atexit(sub_12EDEC0, &qword_4FCED00, &qword_4A427C0);
  qword_4FCEC20 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCEC2C &= 0xF000u;
  qword_4FCEC30 = 0;
  qword_4FCEC38 = 0;
  qword_4FCEC40 = 0;
  qword_4FCEC48 = 0;
  qword_4FCEC50 = 0;
  dword_4FCEC28 = v1;
  qword_4FCEC78 = (__int64)&unk_4FCEC98;
  qword_4FCEC80 = (__int64)&unk_4FCEC98;
  qword_4FCEC68 = (__int64)qword_4FA01C0;
  qword_4FCEC58 = 0;
  qword_4FCECC8 = (__int64)&unk_49E74C8;
  qword_4FCEC60 = 0;
  qword_4FCEC70 = 0;
  qword_4FCEC20 = (__int64)&unk_49EEB70;
  qword_4FCEC88 = 4;
  dword_4FCEC90 = 0;
  qword_4FCECD8 = (__int64)&unk_49EEDF0;
  byte_4FCECB8 = 0;
  dword_4FCECC0 = 0;
  byte_4FCECD4 = 1;
  dword_4FCECD0 = 0;
  sub_16B8280(&qword_4FCEC20, "dfa-sched-reg-pressure-threshold", 32);
  dword_4FCECC0 = 5;
  byte_4FCECD4 = 1;
  dword_4FCECD0 = 5;
  qword_4FCEC50 = 50;
  LOBYTE(word_4FCEC2C) = word_4FCEC2C & 0x98 | 0x21;
  qword_4FCEC48 = (__int64)"Track reg pressure and switch priority to in-depth";
  sub_16B88A0(&qword_4FCEC20);
  return __cxa_atexit(sub_12EDEA0, &qword_4FCEC20, &qword_4A427C0);
}

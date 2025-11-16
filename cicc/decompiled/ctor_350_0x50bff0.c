// Function: ctor_350
// Address: 0x50bff0
//
int ctor_350()
{
  int v0; // edx

  qword_4FCF860 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCF86C &= 0xF000u;
  qword_4FCF870 = 0;
  qword_4FCF8A8 = (__int64)qword_4FA01C0;
  qword_4FCF878 = 0;
  qword_4FCF880 = 0;
  qword_4FCF888 = 0;
  dword_4FCF868 = v0;
  qword_4FCF8B8 = (__int64)&unk_4FCF8D8;
  qword_4FCF8C0 = (__int64)&unk_4FCF8D8;
  qword_4FCF890 = 0;
  qword_4FCF898 = 0;
  qword_4FCF908 = (__int64)&unk_49E74E8;
  word_4FCF910 = 256;
  qword_4FCF8A0 = 0;
  qword_4FCF8B0 = 0;
  qword_4FCF860 = (__int64)&unk_49EEC70;
  qword_4FCF8C8 = 4;
  byte_4FCF8F8 = 0;
  qword_4FCF918 = (__int64)&unk_49EEDB0;
  dword_4FCF8D0 = 0;
  byte_4FCF900 = 0;
  sub_16B8280(&qword_4FCF860, "disable-spill-hoist", 19);
  qword_4FCF890 = 29;
  LOBYTE(word_4FCF86C) = word_4FCF86C & 0x9F | 0x20;
  qword_4FCF888 = (__int64)"Disable inline spill hoisting";
  sub_16B88A0(&qword_4FCF860);
  return __cxa_atexit(sub_12EDEC0, &qword_4FCF860, &qword_4A427C0);
}

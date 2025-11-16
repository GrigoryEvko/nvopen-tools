// Function: ctor_322
// Address: 0x504580
//
int ctor_322()
{
  int v0; // edx

  qword_4FC9CC0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC9CCC &= 0xF000u;
  qword_4FC9CD0 = 0;
  qword_4FC9D08 = (__int64)qword_4FA01C0;
  qword_4FC9CD8 = 0;
  qword_4FC9CE0 = 0;
  qword_4FC9CE8 = 0;
  dword_4FC9CC8 = v0;
  qword_4FC9D18 = (__int64)&unk_4FC9D38;
  qword_4FC9D20 = (__int64)&unk_4FC9D38;
  qword_4FC9CF0 = 0;
  qword_4FC9CF8 = 0;
  qword_4FC9D68 = (__int64)&unk_49E74A8;
  qword_4FC9D00 = 0;
  qword_4FC9D10 = 0;
  qword_4FC9CC0 = (__int64)&unk_49EEAF0;
  qword_4FC9D28 = 4;
  dword_4FC9D30 = 0;
  qword_4FC9D78 = (__int64)&unk_49EEE10;
  byte_4FC9D58 = 0;
  dword_4FC9D60 = 0;
  byte_4FC9D74 = 1;
  dword_4FC9D70 = 0;
  sub_16B8280(&qword_4FC9CC0, "stress-regalloc", 15);
  dword_4FC9D60 = 0;
  byte_4FC9D74 = 1;
  dword_4FC9D70 = 0;
  qword_4FC9D00 = 1;
  LOBYTE(word_4FC9CCC) = word_4FC9CCC & 0x9F | 0x20;
  qword_4FC9CF8 = (__int64)"N";
  qword_4FC9CE8 = (__int64)"Limit all regclasses to N registers";
  qword_4FC9CF0 = 35;
  sub_16B88A0(&qword_4FC9CC0);
  return __cxa_atexit(sub_12EDE60, &qword_4FC9CC0, &qword_4A427C0);
}

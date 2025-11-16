// Function: ctor_227
// Address: 0x4e9e40
//
int ctor_227()
{
  int v0; // edx

  qword_4FB4D80 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB4D8C &= 0xF000u;
  qword_4FB4D90 = 0;
  qword_4FB4DC8 = (__int64)qword_4FA01C0;
  qword_4FB4D98 = 0;
  qword_4FB4DA0 = 0;
  qword_4FB4DA8 = 0;
  dword_4FB4D88 = v0;
  qword_4FB4DD8 = (__int64)&unk_4FB4DF8;
  qword_4FB4DE0 = (__int64)&unk_4FB4DF8;
  qword_4FB4DB0 = 0;
  qword_4FB4DB8 = 0;
  qword_4FB4E28 = (__int64)&unk_49E74E8;
  word_4FB4E30 = 256;
  qword_4FB4DC0 = 0;
  qword_4FB4DD0 = 0;
  qword_4FB4D80 = (__int64)&unk_49EEC70;
  qword_4FB4DE8 = 4;
  byte_4FB4E18 = 0;
  qword_4FB4E38 = (__int64)&unk_49EEDB0;
  dword_4FB4DF0 = 0;
  byte_4FB4E20 = 0;
  sub_16B8280(&qword_4FB4D80, "structurizecfg-skip-uniform-regions", 35);
  word_4FB4E30 = 256;
  byte_4FB4E20 = 0;
  qword_4FB4DB0 = 59;
  LOBYTE(word_4FB4D8C) = word_4FB4D8C & 0x9F | 0x20;
  qword_4FB4DA8 = (__int64)"Force whether the StructurizeCFG pass skips uniform regions";
  sub_16B88A0(&qword_4FB4D80);
  return __cxa_atexit(sub_12EDEC0, &qword_4FB4D80, &qword_4A427C0);
}

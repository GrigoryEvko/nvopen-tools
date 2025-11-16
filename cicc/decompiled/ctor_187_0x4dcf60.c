// Function: ctor_187
// Address: 0x4dcf60
//
int ctor_187()
{
  int v0; // eax
  int v1; // eax

  qword_4FABF20 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FABF2C &= 0xF000u;
  qword_4FABF68 = (__int64)qword_4FA01C0;
  qword_4FABF30 = 0;
  qword_4FABF38 = 0;
  qword_4FABF40 = 0;
  dword_4FABF28 = v0;
  qword_4FABF78 = (__int64)&unk_4FABF98;
  qword_4FABF80 = (__int64)&unk_4FABF98;
  qword_4FABFC0 = (__int64)&byte_4FABFD0;
  qword_4FABFE8 = (__int64)&byte_4FABFF8;
  qword_4FABF48 = 0;
  qword_4FABF50 = 0;
  qword_4FABFE0 = (__int64)&unk_49EED10;
  qword_4FABF58 = 0;
  qword_4FABF60 = 0;
  qword_4FABF20 = (__int64)&unk_49EEBF0;
  qword_4FABF70 = 0;
  qword_4FABF88 = 4;
  qword_4FAC018 = (__int64)&byte_4FAC028;
  qword_4FAC010 = (__int64)&unk_49EEE90;
  dword_4FABF90 = 0;
  byte_4FABFB8 = 0;
  qword_4FABFC8 = 0;
  byte_4FABFD0 = 0;
  qword_4FABFF0 = 0;
  byte_4FABFF8 = 0;
  byte_4FAC008 = 0;
  qword_4FAC020 = 0;
  byte_4FAC028 = 0;
  sub_16B8280(&qword_4FABF20, "internalize-public-api-file", 27);
  qword_4FABF60 = 8;
  qword_4FABF58 = (__int64)"filename";
  qword_4FABF48 = (__int64)"A file containing list of symbol names to preserve";
  qword_4FABF50 = 50;
  sub_16B88A0(&qword_4FABF20);
  __cxa_atexit(sub_12F0C20, &qword_4FABF20, &qword_4A427C0);
  qword_4FABE40 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_4FABE50 = 0;
  qword_4FABE58 = 0;
  qword_4FABE60 = 0;
  qword_4FABE68 = 0;
  qword_4FABE70 = 0;
  qword_4FABE78 = 0;
  dword_4FABE48 = v1;
  qword_4FABE88 = (__int64)qword_4FA01C0;
  qword_4FABF10 = (__int64)&unk_49EEE90;
  qword_4FABE80 = 0;
  word_4FABE4C = word_4FABE4C & 0xF000 | 1;
  qword_4FABE98 = (__int64)&unk_4FABEB8;
  qword_4FABEA0 = (__int64)&unk_4FABEB8;
  qword_4FABE90 = 0;
  byte_4FABED8 = 0;
  qword_4FABE40 = (__int64)&unk_49E75F8;
  qword_4FABEA8 = 4;
  dword_4FABEB0 = 0;
  qword_4FABEE0 = 0;
  qword_4FABEE8 = 0;
  qword_4FABEF0 = 0;
  qword_4FABEF8 = 0;
  qword_4FABF00 = 0;
  qword_4FABF08 = 0;
  sub_16B8280(&qword_4FABE40, "internalize-public-api-list", 27);
  HIBYTE(word_4FABE4C) |= 2u;
  qword_4FABE78 = (__int64)"list";
  qword_4FABE80 = 4;
  qword_4FABE68 = (__int64)"A list of symbol names to preserve";
  qword_4FABE70 = 34;
  sub_16B88A0(&qword_4FABE40);
  return __cxa_atexit(sub_12F08D0, &qword_4FABE40, &qword_4A427C0);
}

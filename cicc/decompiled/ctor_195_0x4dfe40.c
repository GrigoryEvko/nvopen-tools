// Function: ctor_195
// Address: 0x4dfe40
//
int ctor_195()
{
  int v0; // eax
  int v1; // eax

  qword_4FAE0C0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAE0CC &= 0xF000u;
  qword_4FAE0D0 = 0;
  qword_4FAE0D8 = 0;
  qword_4FAE0E0 = 0;
  qword_4FAE0E8 = 0;
  qword_4FAE0F0 = 0;
  dword_4FAE0C8 = v0;
  qword_4FAE0F8 = 0;
  qword_4FAE108 = (__int64)qword_4FA01C0;
  qword_4FAE118 = (__int64)&unk_4FAE138;
  qword_4FAE120 = (__int64)&unk_4FAE138;
  qword_4FAE100 = 0;
  qword_4FAE110 = 0;
  word_4FAE170 = 256;
  qword_4FAE168 = (__int64)&unk_49E74E8;
  qword_4FAE128 = 4;
  qword_4FAE0C0 = (__int64)&unk_49EEC70;
  byte_4FAE158 = 0;
  qword_4FAE178 = (__int64)&unk_49EEDB0;
  dword_4FAE130 = 0;
  byte_4FAE160 = 0;
  sub_16B8280(&qword_4FAE0C0, "adce-remove-control-flow", 24);
  word_4FAE170 = 257;
  byte_4FAE160 = 1;
  LOBYTE(word_4FAE0CC) = word_4FAE0CC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FAE0C0);
  __cxa_atexit(sub_12EDEC0, &qword_4FAE0C0, &qword_4A427C0);
  qword_4FADFE0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAE090 = 256;
  word_4FADFEC &= 0xF000u;
  qword_4FADFF0 = 0;
  qword_4FADFF8 = 0;
  qword_4FAE000 = 0;
  dword_4FADFE8 = v1;
  qword_4FAE088 = (__int64)&unk_49E74E8;
  qword_4FAE028 = (__int64)qword_4FA01C0;
  qword_4FAE038 = (__int64)&unk_4FAE058;
  qword_4FAE040 = (__int64)&unk_4FAE058;
  qword_4FADFE0 = (__int64)&unk_49EEC70;
  qword_4FAE098 = (__int64)&unk_49EEDB0;
  qword_4FAE008 = 0;
  qword_4FAE010 = 0;
  qword_4FAE018 = 0;
  qword_4FAE020 = 0;
  qword_4FAE030 = 0;
  qword_4FAE048 = 4;
  dword_4FAE050 = 0;
  byte_4FAE078 = 0;
  byte_4FAE080 = 0;
  sub_16B8280(&qword_4FADFE0, "adce-remove-loops", 17);
  word_4FAE090 = 256;
  byte_4FAE080 = 0;
  LOBYTE(word_4FADFEC) = word_4FADFEC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FADFE0);
  return __cxa_atexit(sub_12EDEC0, &qword_4FADFE0, &qword_4A427C0);
}

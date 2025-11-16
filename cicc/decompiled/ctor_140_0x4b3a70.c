// Function: ctor_140
// Address: 0x4b3a70
//
int ctor_140()
{
  int v0; // edx

  qword_4F9DB20 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9DB2C &= 0xF000u;
  qword_4F9DB30 = 0;
  qword_4F9DB68 = (__int64)&unk_4FA01C0;
  qword_4F9DB38 = 0;
  qword_4F9DB40 = 0;
  qword_4F9DB48 = 0;
  dword_4F9DB28 = v0;
  qword_4F9DB78 = (__int64)&unk_4F9DB98;
  qword_4F9DB80 = (__int64)&unk_4F9DB98;
  qword_4F9DB50 = 0;
  qword_4F9DB58 = 0;
  qword_4F9DBC8 = (__int64)&unk_49E74E8;
  word_4F9DBD0 = 256;
  qword_4F9DB60 = 0;
  qword_4F9DB70 = 0;
  qword_4F9DB20 = (__int64)&unk_49EEC70;
  qword_4F9DB88 = 4;
  byte_4F9DBB8 = 0;
  qword_4F9DBD8 = (__int64)&unk_49EEDB0;
  dword_4F9DB90 = 0;
  byte_4F9DBC0 = 0;
  sub_16B8280(&qword_4F9DB20, "print-summary-global-ids", 24);
  word_4F9DBD0 = 256;
  byte_4F9DBC0 = 0;
  qword_4F9DB50 = 66;
  LOBYTE(word_4F9DB2C) = word_4F9DB2C & 0x9F | 0x20;
  qword_4F9DB48 = (__int64)"Print the global id for each value when reading the module summary";
  sub_16B88A0(&qword_4F9DB20);
  return __cxa_atexit(sub_12EDEC0, &qword_4F9DB20, &qword_4A427C0);
}

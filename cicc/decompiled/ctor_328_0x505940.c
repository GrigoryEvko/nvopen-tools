// Function: ctor_328
// Address: 0x505940
//
int ctor_328()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax

  qword_4FCAA00 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCAA0C &= 0xF000u;
  qword_4FCAA10 = 0;
  qword_4FCAA18 = 0;
  qword_4FCAA20 = 0;
  qword_4FCAA28 = 0;
  qword_4FCAA30 = 0;
  dword_4FCAA08 = v0;
  qword_4FCAA38 = 0;
  qword_4FCAA48 = (__int64)qword_4FA01C0;
  qword_4FCAA58 = (__int64)&unk_4FCAA78;
  qword_4FCAA60 = (__int64)&unk_4FCAA78;
  qword_4FCAA40 = 0;
  qword_4FCAA50 = 0;
  word_4FCAAB0 = 256;
  qword_4FCAAA8 = (__int64)&unk_49E74E8;
  qword_4FCAA68 = 4;
  qword_4FCAA00 = (__int64)&unk_49EEC70;
  byte_4FCAA98 = 0;
  qword_4FCAAB8 = (__int64)&unk_49EEDB0;
  dword_4FCAA70 = 0;
  byte_4FCAAA0 = 0;
  sub_16B8280(&qword_4FCAA00, "no-stack-coloring", 17);
  word_4FCAAB0 = 256;
  byte_4FCAAA0 = 0;
  qword_4FCAA30 = 22;
  LOBYTE(word_4FCAA0C) = word_4FCAA0C & 0x9F | 0x20;
  qword_4FCAA28 = (__int64)"Disable stack coloring";
  sub_16B88A0(&qword_4FCAA00);
  __cxa_atexit(sub_12EDEC0, &qword_4FCAA00, &qword_4A427C0);
  qword_4FCA920 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCA9D0 = 256;
  word_4FCA92C &= 0xF000u;
  qword_4FCA930 = 0;
  qword_4FCA938 = 0;
  qword_4FCA940 = 0;
  dword_4FCA928 = v1;
  qword_4FCA9C8 = (__int64)&unk_49E74E8;
  qword_4FCA968 = (__int64)qword_4FA01C0;
  qword_4FCA978 = (__int64)&unk_4FCA998;
  qword_4FCA980 = (__int64)&unk_4FCA998;
  qword_4FCA920 = (__int64)&unk_49EEC70;
  qword_4FCA9D8 = (__int64)&unk_49EEDB0;
  qword_4FCA948 = 0;
  qword_4FCA950 = 0;
  qword_4FCA958 = 0;
  qword_4FCA960 = 0;
  qword_4FCA970 = 0;
  qword_4FCA988 = 4;
  dword_4FCA990 = 0;
  byte_4FCA9B8 = 0;
  byte_4FCA9C0 = 0;
  sub_16B8280(&qword_4FCA920, "protect-from-escaped-allocas", 28);
  word_4FCA9D0 = 256;
  byte_4FCA9C0 = 0;
  qword_4FCA950 = 46;
  LOBYTE(word_4FCA92C) = word_4FCA92C & 0x9F | 0x20;
  qword_4FCA948 = (__int64)"Do not optimize lifetime zones that are broken";
  sub_16B88A0(&qword_4FCA920);
  __cxa_atexit(sub_12EDEC0, &qword_4FCA920, &qword_4A427C0);
  qword_4FCA840 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCA8F0 = 256;
  word_4FCA84C &= 0xF000u;
  qword_4FCA850 = 0;
  qword_4FCA858 = 0;
  qword_4FCA860 = 0;
  dword_4FCA848 = v2;
  qword_4FCA8E8 = (__int64)&unk_49E74E8;
  qword_4FCA888 = (__int64)qword_4FA01C0;
  qword_4FCA898 = (__int64)&unk_4FCA8B8;
  qword_4FCA8A0 = (__int64)&unk_4FCA8B8;
  qword_4FCA840 = (__int64)&unk_49EEC70;
  qword_4FCA8F8 = (__int64)&unk_49EEDB0;
  qword_4FCA868 = 0;
  qword_4FCA870 = 0;
  qword_4FCA878 = 0;
  qword_4FCA880 = 0;
  qword_4FCA890 = 0;
  qword_4FCA8A8 = 4;
  dword_4FCA8B0 = 0;
  byte_4FCA8D8 = 0;
  byte_4FCA8E0 = 0;
  sub_16B8280(&qword_4FCA840, "stackcoloring-lifetime-start-on-first-use", 41);
  byte_4FCA8E0 = 1;
  word_4FCA8F0 = 257;
  qword_4FCA870 = 68;
  LOBYTE(word_4FCA84C) = word_4FCA84C & 0x9F | 0x20;
  qword_4FCA868 = (__int64)"Treat stack lifetimes as starting on first use, not on START marker.";
  sub_16B88A0(&qword_4FCA840);
  return __cxa_atexit(sub_12EDEC0, &qword_4FCA840, &qword_4A427C0);
}

// Function: ctor_333
// Address: 0x5069e0
//
int ctor_333()
{
  int v0; // edx

  qword_4FCB1E0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCB1EC &= 0xF000u;
  qword_4FCB1F0 = 0;
  qword_4FCB228 = (__int64)qword_4FA01C0;
  qword_4FCB1F8 = 0;
  qword_4FCB200 = 0;
  qword_4FCB208 = 0;
  dword_4FCB1E8 = v0;
  qword_4FCB238 = (__int64)&unk_4FCB258;
  qword_4FCB240 = (__int64)&unk_4FCB258;
  qword_4FCB210 = 0;
  qword_4FCB218 = 0;
  qword_4FCB288 = (__int64)&unk_49E74E8;
  word_4FCB290 = 256;
  qword_4FCB220 = 0;
  qword_4FCB230 = 0;
  qword_4FCB1E0 = (__int64)&unk_49EEC70;
  qword_4FCB248 = 4;
  byte_4FCB278 = 0;
  qword_4FCB298 = (__int64)&unk_49EEDB0;
  dword_4FCB250 = 0;
  byte_4FCB280 = 0;
  sub_16B8280(&qword_4FCB1E0, "disable-sched-hazard", 20);
  word_4FCB290 = 256;
  byte_4FCB280 = 0;
  qword_4FCB210 = 48;
  LOBYTE(word_4FCB1EC) = word_4FCB1EC & 0x9F | 0x20;
  qword_4FCB208 = (__int64)"Disable hazard detection during preRA scheduling";
  sub_16B88A0(&qword_4FCB1E0);
  return __cxa_atexit(sub_12EDEC0, &qword_4FCB1E0, &qword_4A427C0);
}

// Function: ctor_348
// Address: 0x50bcb0
//
int ctor_348()
{
  int v0; // edx

  qword_4FCF6A0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCF6AC &= 0xF000u;
  qword_4FCF6B0 = 0;
  qword_4FCF6E8 = (__int64)qword_4FA01C0;
  qword_4FCF6B8 = 0;
  qword_4FCF6C0 = 0;
  qword_4FCF6C8 = 0;
  dword_4FCF6A8 = v0;
  qword_4FCF6F8 = (__int64)&unk_4FCF718;
  qword_4FCF700 = (__int64)&unk_4FCF718;
  qword_4FCF6D0 = 0;
  qword_4FCF6D8 = 0;
  qword_4FCF748 = (__int64)&unk_49E74A8;
  qword_4FCF6E0 = 0;
  qword_4FCF6F0 = 0;
  qword_4FCF6A0 = (__int64)&unk_49EEAF0;
  qword_4FCF708 = 4;
  dword_4FCF710 = 0;
  qword_4FCF758 = (__int64)&unk_49EEE10;
  byte_4FCF738 = 0;
  dword_4FCF740 = 0;
  byte_4FCF754 = 1;
  dword_4FCF750 = 0;
  sub_16B8280(&qword_4FCF6A0, "dfa-instr-limit", 15);
  dword_4FCF740 = 0;
  byte_4FCF754 = 1;
  dword_4FCF750 = 0;
  qword_4FCF6D0 = 50;
  LOBYTE(word_4FCF6AC) = word_4FCF6AC & 0x9F | 0x20;
  qword_4FCF6C8 = (__int64)"If present, stops packetizing after N instructions";
  sub_16B88A0(&qword_4FCF6A0);
  return __cxa_atexit(sub_12EDE60, &qword_4FCF6A0, &qword_4A427C0);
}

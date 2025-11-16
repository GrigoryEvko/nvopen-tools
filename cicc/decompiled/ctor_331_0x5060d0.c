// Function: ctor_331
// Address: 0x5060d0
//
int ctor_331()
{
  int v0; // eax
  int v1; // eax

  qword_4FCAD80 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCAD8C &= 0xF000u;
  qword_4FCADC8 = (__int64)qword_4FA01C0;
  qword_4FCAD90 = 0;
  qword_4FCAD98 = 0;
  qword_4FCADA0 = 0;
  dword_4FCAD88 = v0;
  qword_4FCADD8 = (__int64)&unk_4FCADF8;
  qword_4FCADE0 = (__int64)&unk_4FCADF8;
  qword_4FCADA8 = 0;
  qword_4FCADB0 = 0;
  qword_4FCAE28 = (__int64)&unk_49E74E8;
  word_4FCAE30 = 256;
  qword_4FCADB8 = 0;
  qword_4FCADC0 = 0;
  qword_4FCAD80 = (__int64)&unk_49EEC70;
  qword_4FCADD0 = 0;
  byte_4FCAE18 = 0;
  qword_4FCAE38 = (__int64)&unk_49EEDB0;
  qword_4FCADE8 = 4;
  dword_4FCADF0 = 0;
  byte_4FCAE20 = 0;
  sub_16B8280(&qword_4FCAD80, "no-stack-slot-sharing", 21);
  word_4FCAE30 = 256;
  byte_4FCAE20 = 0;
  qword_4FCADB0 = 43;
  LOBYTE(word_4FCAD8C) = word_4FCAD8C & 0x9F | 0x20;
  qword_4FCADA8 = (__int64)"Suppress slot sharing during stack coloring";
  sub_16B88A0(&qword_4FCAD80);
  __cxa_atexit(sub_12EDEC0, &qword_4FCAD80, &qword_4A427C0);
  qword_4FCACA0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCACAC &= 0xF000u;
  qword_4FCACB0 = 0;
  qword_4FCACB8 = 0;
  qword_4FCACC0 = 0;
  qword_4FCACC8 = 0;
  qword_4FCACD0 = 0;
  dword_4FCACA8 = v1;
  qword_4FCACF8 = (__int64)&unk_4FCAD18;
  qword_4FCAD00 = (__int64)&unk_4FCAD18;
  qword_4FCACE8 = (__int64)qword_4FA01C0;
  qword_4FCACD8 = 0;
  qword_4FCAD48 = (__int64)&unk_49E74C8;
  qword_4FCACE0 = 0;
  qword_4FCACF0 = 0;
  qword_4FCACA0 = (__int64)&unk_49EEB70;
  qword_4FCAD08 = 4;
  dword_4FCAD10 = 0;
  qword_4FCAD58 = (__int64)&unk_49EEDF0;
  byte_4FCAD38 = 0;
  dword_4FCAD40 = 0;
  byte_4FCAD54 = 1;
  dword_4FCAD50 = 0;
  sub_16B8280(&qword_4FCACA0, "ssc-dce-limit", 13);
  dword_4FCAD40 = -1;
  byte_4FCAD54 = 1;
  dword_4FCAD50 = -1;
  LOBYTE(word_4FCACAC) = word_4FCACAC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FCACA0);
  return __cxa_atexit(sub_12EDEA0, &qword_4FCACA0, &qword_4A427C0);
}

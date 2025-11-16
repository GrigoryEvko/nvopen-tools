// Function: ctor_359
// Address: 0x510590
//
int ctor_359()
{
  int v0; // edx

  qword_4FD2980 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FD298C &= 0xF000u;
  qword_4FD2990 = 0;
  qword_4FD29C8 = (__int64)qword_4FA01C0;
  qword_4FD2998 = 0;
  qword_4FD29A0 = 0;
  qword_4FD29A8 = 0;
  dword_4FD2988 = v0;
  qword_4FD29D8 = (__int64)&unk_4FD29F8;
  qword_4FD29E0 = (__int64)&unk_4FD29F8;
  qword_4FD29B0 = 0;
  qword_4FD29B8 = 0;
  qword_4FD2A28 = (__int64)&unk_49E74A8;
  qword_4FD29C0 = 0;
  qword_4FD29D0 = 0;
  qword_4FD2980 = (__int64)&unk_49EEAF0;
  qword_4FD29E8 = 4;
  dword_4FD29F0 = 0;
  qword_4FD2A38 = (__int64)&unk_49EEE10;
  byte_4FD2A18 = 0;
  dword_4FD2A20 = 0;
  byte_4FD2A34 = 1;
  dword_4FD2A30 = 0;
  sub_16B8280(&qword_4FD2980, "nvvm-intr-range-sm", 18);
  dword_4FD2A20 = 20;
  byte_4FD2A34 = 1;
  dword_4FD2A30 = 20;
  qword_4FD29B0 = 10;
  LOBYTE(word_4FD298C) = word_4FD298C & 0x9F | 0x20;
  qword_4FD29A8 = (__int64)"SM variant";
  sub_16B88A0(&qword_4FD2980);
  return __cxa_atexit(sub_12EDE60, &qword_4FD2980, &qword_4A427C0);
}

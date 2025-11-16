// Function: ctor_217
// Address: 0x4e7700
//
int ctor_217()
{
  int v0; // eax
  int v1; // eax

  qword_4FB3680 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB368C &= 0xF000u;
  qword_4FB3690 = 0;
  qword_4FB3698 = 0;
  qword_4FB36A0 = 0;
  qword_4FB36A8 = 0;
  qword_4FB36B0 = 0;
  dword_4FB3688 = v0;
  qword_4FB36B8 = 0;
  qword_4FB36C8 = (__int64)qword_4FA01C0;
  qword_4FB36D8 = (__int64)&unk_4FB36F8;
  qword_4FB36E0 = (__int64)&unk_4FB36F8;
  qword_4FB36C0 = 0;
  qword_4FB36D0 = 0;
  qword_4FB3728 = (__int64)&unk_49E74A8;
  qword_4FB36E8 = 4;
  qword_4FB3680 = (__int64)&unk_49EEAF0;
  dword_4FB36F0 = 0;
  qword_4FB3738 = (__int64)&unk_49EEE10;
  byte_4FB3718 = 0;
  dword_4FB3720 = 0;
  byte_4FB3734 = 1;
  dword_4FB3730 = 0;
  sub_16B8280(&qword_4FB3680, "max-switch-cases", 16);
  qword_4FB36B0 = 100;
  qword_4FB36A8 = (__int64)"Max switch cases for fully unrolled loops where we decide to unswitch without checking profitability";
  dword_4FB3720 = 4;
  byte_4FB3734 = 1;
  dword_4FB3730 = 4;
  LOBYTE(word_4FB368C) = word_4FB368C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FB3680);
  __cxa_atexit(sub_12EDE60, &qword_4FB3680, &qword_4A427C0);
  qword_4FB35A0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB35AC &= 0xF000u;
  qword_4FB35B0 = 0;
  qword_4FB35B8 = 0;
  qword_4FB35C0 = 0;
  qword_4FB35C8 = 0;
  qword_4FB35D0 = 0;
  dword_4FB35A8 = v1;
  qword_4FB3648 = (__int64)&unk_49E74A8;
  qword_4FB35E8 = (__int64)qword_4FA01C0;
  qword_4FB35F8 = (__int64)&unk_4FB3618;
  qword_4FB3600 = (__int64)&unk_4FB3618;
  qword_4FB35A0 = (__int64)&unk_49EEAF0;
  qword_4FB3658 = (__int64)&unk_49EEE10;
  qword_4FB35D8 = 0;
  qword_4FB35E0 = 0;
  qword_4FB35F0 = 0;
  qword_4FB3608 = 4;
  dword_4FB3610 = 0;
  byte_4FB3638 = 0;
  dword_4FB3640 = 0;
  byte_4FB3654 = 1;
  dword_4FB3650 = 0;
  sub_16B8280(&qword_4FB35A0, "loop-unswitch-threshold", 23);
  qword_4FB35D0 = 25;
  qword_4FB35C8 = (__int64)"Max loop size to unswitch";
  dword_4FB3640 = 150;
  byte_4FB3654 = 1;
  dword_4FB3650 = 150;
  LOBYTE(word_4FB35AC) = word_4FB35AC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FB35A0);
  return __cxa_atexit(sub_12EDE60, &qword_4FB35A0, &qword_4A427C0);
}

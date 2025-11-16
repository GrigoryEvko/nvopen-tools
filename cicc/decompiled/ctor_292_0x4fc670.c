// Function: ctor_292
// Address: 0x4fc670
//
int ctor_292()
{
  int v0; // eax
  int v1; // eax

  qword_4FC41A0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC41AC &= 0xF000u;
  qword_4FC41E8 = (__int64)qword_4FA01C0;
  qword_4FC41B0 = 0;
  qword_4FC41B8 = 0;
  qword_4FC41C0 = 0;
  dword_4FC41A8 = v0;
  qword_4FC41F8 = (__int64)&unk_4FC4218;
  qword_4FC4200 = (__int64)&unk_4FC4218;
  qword_4FC41C8 = 0;
  qword_4FC41D0 = 0;
  qword_4FC4248 = (__int64)&unk_49E74C8;
  qword_4FC41D8 = 0;
  qword_4FC41E0 = 0;
  qword_4FC41A0 = (__int64)&unk_49EEB70;
  qword_4FC41F0 = 0;
  byte_4FC4238 = 0;
  qword_4FC4258 = (__int64)&unk_49EEDF0;
  qword_4FC4208 = 4;
  dword_4FC4210 = 0;
  dword_4FC4240 = 0;
  byte_4FC4254 = 1;
  dword_4FC4250 = 0;
  sub_16B8280(&qword_4FC41A0, "imp-null-check-page-size", 24);
  qword_4FC41D0 = 36;
  qword_4FC41C8 = (__int64)"The page size of the target in bytes";
  dword_4FC4240 = 4096;
  byte_4FC4254 = 1;
  dword_4FC4250 = 4096;
  LOBYTE(word_4FC41AC) = word_4FC41AC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC41A0);
  __cxa_atexit(sub_12EDEA0, &qword_4FC41A0, &qword_4A427C0);
  qword_4FC40C0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC40CC &= 0xF000u;
  qword_4FC40D0 = 0;
  qword_4FC40D8 = 0;
  qword_4FC40E0 = 0;
  qword_4FC40E8 = 0;
  qword_4FC40F0 = 0;
  dword_4FC40C8 = v1;
  qword_4FC4118 = (__int64)&unk_4FC4138;
  qword_4FC4120 = (__int64)&unk_4FC4138;
  qword_4FC4108 = (__int64)qword_4FA01C0;
  qword_4FC40F8 = 0;
  qword_4FC4168 = (__int64)&unk_49E74A8;
  qword_4FC4100 = 0;
  qword_4FC4110 = 0;
  qword_4FC40C0 = (__int64)&unk_49EEAF0;
  qword_4FC4128 = 4;
  dword_4FC4130 = 0;
  qword_4FC4178 = (__int64)&unk_49EEE10;
  byte_4FC4158 = 0;
  dword_4FC4160 = 0;
  byte_4FC4174 = 1;
  dword_4FC4170 = 0;
  sub_16B8280(&qword_4FC40C0, "imp-null-max-insts-to-consider", 30);
  qword_4FC40F0 = 108;
  qword_4FC40E8 = (__int64)"The max number of instructions to consider hoisting loads over (the algorithm is quadratic over this number)";
  dword_4FC4160 = 8;
  byte_4FC4174 = 1;
  dword_4FC4170 = 8;
  LOBYTE(word_4FC40CC) = word_4FC40CC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC40C0);
  return __cxa_atexit(sub_12EDE60, &qword_4FC40C0, &qword_4A427C0);
}

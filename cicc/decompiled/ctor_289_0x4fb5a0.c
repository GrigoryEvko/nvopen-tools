// Function: ctor_289
// Address: 0x4fb5a0
//
int ctor_289()
{
  int v0; // eax
  int v1; // eax

  qword_4FC3440 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC344C &= 0xF000u;
  qword_4FC3488 = (__int64)qword_4FA01C0;
  qword_4FC3450 = 0;
  qword_4FC3458 = 0;
  qword_4FC3460 = 0;
  dword_4FC3448 = v0;
  qword_4FC3498 = (__int64)&unk_4FC34B8;
  qword_4FC34A0 = (__int64)&unk_4FC34B8;
  qword_4FC3468 = 0;
  qword_4FC3470 = 0;
  qword_4FC34E8 = (__int64)&unk_49E74A8;
  qword_4FC3478 = 0;
  qword_4FC3480 = 0;
  qword_4FC3440 = (__int64)&unk_49EEAF0;
  qword_4FC3490 = 0;
  byte_4FC34D8 = 0;
  qword_4FC34F8 = (__int64)&unk_49EEE10;
  qword_4FC34A8 = 4;
  dword_4FC34B0 = 0;
  dword_4FC34E0 = 0;
  byte_4FC34F4 = 1;
  dword_4FC34F0 = 0;
  sub_16B8280(&qword_4FC3440, "early-ifcvt-limit", 17);
  dword_4FC34E0 = 30;
  byte_4FC34F4 = 1;
  dword_4FC34F0 = 30;
  qword_4FC3470 = 52;
  LOBYTE(word_4FC344C) = word_4FC344C & 0x9F | 0x20;
  qword_4FC3468 = (__int64)"Maximum number of instructions per speculated block.";
  sub_16B88A0(&qword_4FC3440);
  __cxa_atexit(sub_12EDE60, &qword_4FC3440, &qword_4A427C0);
  qword_4FC3360 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC336C &= 0xF000u;
  qword_4FC3370 = 0;
  qword_4FC3378 = 0;
  qword_4FC3380 = 0;
  qword_4FC3388 = 0;
  qword_4FC3390 = 0;
  dword_4FC3368 = v1;
  qword_4FC33B8 = (__int64)&unk_4FC33D8;
  qword_4FC33C0 = (__int64)&unk_4FC33D8;
  qword_4FC33A8 = (__int64)qword_4FA01C0;
  qword_4FC3398 = 0;
  qword_4FC3408 = (__int64)&unk_49E74E8;
  word_4FC3410 = 256;
  qword_4FC33A0 = 0;
  qword_4FC33B0 = 0;
  qword_4FC3360 = (__int64)&unk_49EEC70;
  qword_4FC33C8 = 4;
  byte_4FC33F8 = 0;
  qword_4FC3418 = (__int64)&unk_49EEDB0;
  dword_4FC33D0 = 0;
  byte_4FC3400 = 0;
  sub_16B8280(&qword_4FC3360, "stress-early-ifcvt", 18);
  qword_4FC3390 = 20;
  LOBYTE(word_4FC336C) = word_4FC336C & 0x9F | 0x20;
  qword_4FC3388 = (__int64)"Turn all knobs to 11";
  sub_16B88A0(&qword_4FC3360);
  return __cxa_atexit(sub_12EDEC0, &qword_4FC3360, &qword_4A427C0);
}

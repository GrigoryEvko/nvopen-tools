// Function: ctor_313
// Address: 0x5021c0
//
int ctor_313()
{
  int v0; // edx

  qword_4FC8400 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC840C &= 0xF000u;
  qword_4FC8410 = 0;
  qword_4FC8448 = (__int64)qword_4FA01C0;
  qword_4FC8418 = 0;
  qword_4FC8420 = 0;
  qword_4FC8428 = 0;
  dword_4FC8408 = v0;
  qword_4FC8458 = (__int64)&unk_4FC8478;
  qword_4FC8460 = (__int64)&unk_4FC8478;
  qword_4FC8430 = 0;
  qword_4FC8438 = 0;
  qword_4FC84A8 = (__int64)&unk_49E74C8;
  qword_4FC8440 = 0;
  qword_4FC8450 = 0;
  qword_4FC8400 = (__int64)&unk_49EEB70;
  qword_4FC8468 = 4;
  dword_4FC8470 = 0;
  qword_4FC84B8 = (__int64)&unk_49EEDF0;
  byte_4FC8498 = 0;
  dword_4FC84A0 = 0;
  byte_4FC84B4 = 1;
  dword_4FC84B0 = 0;
  sub_16B8280(&qword_4FC8400, "hoistphiconsts", 14);
  qword_4FC8430 = 20;
  qword_4FC8428 = (__int64)"Hoist consts in phis";
  dword_4FC84A0 = 1;
  byte_4FC84B4 = 1;
  dword_4FC84B0 = 1;
  sub_16B88A0(&qword_4FC8400);
  return __cxa_atexit(sub_12EDEA0, &qword_4FC8400, &qword_4A427C0);
}

// Function: ctor_268
// Address: 0x4f62d0
//
int ctor_268()
{
  int v0; // eax
  int v1; // eax

  qword_4FBE660 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FBE66C &= 0xF000u;
  qword_4FBE6A8 = (__int64)qword_4FA01C0;
  qword_4FBE670 = 0;
  qword_4FBE678 = 0;
  qword_4FBE680 = 0;
  dword_4FBE668 = v0;
  qword_4FBE6B8 = (__int64)&unk_4FBE6D8;
  qword_4FBE6C0 = (__int64)&unk_4FBE6D8;
  qword_4FBE688 = 0;
  qword_4FBE690 = 0;
  qword_4FBE708 = (__int64)&unk_49E74C8;
  qword_4FBE698 = 0;
  qword_4FBE6A0 = 0;
  qword_4FBE660 = (__int64)&unk_49EEB70;
  qword_4FBE6B0 = 0;
  byte_4FBE6F8 = 0;
  qword_4FBE718 = (__int64)&unk_49EEDF0;
  qword_4FBE6C8 = 4;
  dword_4FBE6D0 = 0;
  dword_4FBE700 = 0;
  byte_4FBE714 = 1;
  dword_4FBE710 = 0;
  sub_16B8280(&qword_4FBE660, "normalize-gep", 13);
  dword_4FBE700 = 1;
  byte_4FBE714 = 1;
  dword_4FBE710 = 1;
  qword_4FBE690 = 31;
  LOBYTE(word_4FBE66C) = word_4FBE66C & 0x9F | 0x20;
  qword_4FBE688 = (__int64)"Normalize 64-bit GEP subscripts";
  sub_16B88A0(&qword_4FBE660);
  __cxa_atexit(sub_12EDEA0, &qword_4FBE660, &qword_4A427C0);
  qword_4FBE580 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FBE58C &= 0xF000u;
  qword_4FBE590 = 0;
  qword_4FBE598 = 0;
  qword_4FBE5A0 = 0;
  qword_4FBE5A8 = 0;
  qword_4FBE5B0 = 0;
  dword_4FBE588 = v1;
  qword_4FBE5D8 = (__int64)&unk_4FBE5F8;
  qword_4FBE5E0 = (__int64)&unk_4FBE5F8;
  qword_4FBE5C8 = (__int64)qword_4FA01C0;
  qword_4FBE5B8 = 0;
  qword_4FBE628 = (__int64)&unk_49E74E8;
  word_4FBE630 = 256;
  qword_4FBE5C0 = 0;
  qword_4FBE5D0 = 0;
  qword_4FBE580 = (__int64)&unk_49EEC70;
  qword_4FBE5E8 = 4;
  byte_4FBE618 = 0;
  qword_4FBE638 = (__int64)&unk_49EEDB0;
  dword_4FBE5F0 = 0;
  byte_4FBE620 = 0;
  sub_16B8280(&qword_4FBE580, "dump-normalize-gep", 18);
  word_4FBE630 = 256;
  byte_4FBE620 = 0;
  qword_4FBE5B0 = 57;
  LOBYTE(word_4FBE58C) = word_4FBE58C & 0x9F | 0x20;
  qword_4FBE5A8 = (__int64)"Dump Debug Message during Normalize 64-bit GEP subscripts";
  sub_16B88A0(&qword_4FBE580);
  return __cxa_atexit(sub_12EDEC0, &qword_4FBE580, &qword_4A427C0);
}

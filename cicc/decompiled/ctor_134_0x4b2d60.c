// Function: ctor_134
// Address: 0x4b2d60
//
int ctor_134()
{
  int v0; // edx

  qword_4F9D3E0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9D3EC &= 0xF000u;
  qword_4F9D3F0 = 0;
  qword_4F9D428 = (__int64)&unk_4FA01C0;
  qword_4F9D3F8 = 0;
  qword_4F9D400 = 0;
  qword_4F9D408 = 0;
  dword_4F9D3E8 = v0;
  qword_4F9D438 = (__int64)&unk_4F9D458;
  qword_4F9D440 = (__int64)&unk_4F9D458;
  qword_4F9D410 = 0;
  qword_4F9D418 = 0;
  qword_4F9D488 = (__int64)&unk_49E74E8;
  word_4F9D490 = 256;
  qword_4F9D420 = 0;
  qword_4F9D430 = 0;
  qword_4F9D3E0 = (__int64)&unk_49EEC70;
  qword_4F9D448 = 4;
  byte_4F9D478 = 0;
  qword_4F9D498 = (__int64)&unk_49EEDB0;
  dword_4F9D450 = 0;
  byte_4F9D480 = 0;
  sub_16B8280(&qword_4F9D3E0, "costmodel-reduxcost", 19);
  word_4F9D490 = 256;
  byte_4F9D480 = 0;
  qword_4F9D410 = 29;
  LOBYTE(word_4F9D3EC) = word_4F9D3EC & 0x9F | 0x20;
  qword_4F9D408 = (__int64)"Recognize reduction patterns.";
  sub_16B88A0(&qword_4F9D3E0);
  return __cxa_atexit(sub_12EDEC0, &qword_4F9D3E0, &qword_4A427C0);
}

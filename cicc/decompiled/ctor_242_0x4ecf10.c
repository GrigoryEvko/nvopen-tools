// Function: ctor_242
// Address: 0x4ecf10
//
int ctor_242()
{
  int v0; // edx

  qword_4FB6D20 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB6D2C &= 0xF000u;
  qword_4FB6D30 = 0;
  qword_4FB6D68 = (__int64)qword_4FA01C0;
  qword_4FB6D38 = 0;
  qword_4FB6D40 = 0;
  qword_4FB6D48 = 0;
  dword_4FB6D28 = v0;
  qword_4FB6D78 = (__int64)&unk_4FB6D98;
  qword_4FB6D80 = (__int64)&unk_4FB6D98;
  qword_4FB6D50 = 0;
  qword_4FB6D58 = 0;
  qword_4FB6DC8 = (__int64)&unk_49E74A8;
  qword_4FB6D60 = 0;
  qword_4FB6D70 = 0;
  qword_4FB6D20 = (__int64)&unk_49EEAF0;
  qword_4FB6D88 = 4;
  dword_4FB6D90 = 0;
  qword_4FB6DD8 = (__int64)&unk_49EEE10;
  byte_4FB6DB8 = 0;
  dword_4FB6DC0 = 0;
  byte_4FB6DD4 = 1;
  dword_4FB6DD0 = 0;
  sub_16B8280(&qword_4FB6D20, "max-mem2reg-size", 16);
  qword_4FB6D50 = 43;
  dword_4FB6DC0 = 64;
  byte_4FB6DD4 = 1;
  dword_4FB6DD0 = 64;
  LOBYTE(word_4FB6D2C) = word_4FB6D2C & 0x9F | 0x20;
  qword_4FB6D48 = (__int64)"Maximum size in bits of a registrable value";
  sub_16B88A0(&qword_4FB6D20);
  return __cxa_atexit(sub_12EDE60, &qword_4FB6D20, &qword_4A427C0);
}

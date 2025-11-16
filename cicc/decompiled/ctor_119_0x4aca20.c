// Function: ctor_119
// Address: 0x4aca20
//
int ctor_119()
{
  int v0; // edx

  qword_4F98D40 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F98D4C &= 0xF000u;
  qword_4F98D50 = 0;
  qword_4F98D88 = (__int64)&unk_4FA01C0;
  qword_4F98D58 = 0;
  qword_4F98D60 = 0;
  qword_4F98D68 = 0;
  dword_4F98D48 = v0;
  qword_4F98D98 = (__int64)&unk_4F98DB8;
  qword_4F98DA0 = (__int64)&unk_4F98DB8;
  qword_4F98D70 = 0;
  qword_4F98D78 = 0;
  qword_4F98DE8 = (__int64)&unk_49E74E8;
  word_4F98DF0 = 256;
  qword_4F98D80 = 0;
  qword_4F98D90 = 0;
  qword_4F98D40 = (__int64)&unk_49EEC70;
  qword_4F98DA8 = 4;
  byte_4F98DD8 = 0;
  qword_4F98DF8 = (__int64)&unk_49EEDB0;
  dword_4F98DB0 = 0;
  byte_4F98DE0 = 0;
  sub_16B8280(&qword_4F98D40, "da-delinearize", 14);
  word_4F98DF0 = 257;
  byte_4F98DE0 = 1;
  qword_4F98D70 = 36;
  LOBYTE(word_4F98D4C) = word_4F98D4C & 0x98 | 0x21;
  qword_4F98D68 = (__int64)"Try to delinearize array references.";
  sub_16B88A0(&qword_4F98D40);
  return __cxa_atexit(sub_12EDEC0, &qword_4F98D40, &qword_4A427C0);
}

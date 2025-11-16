// Function: ctor_244
// Address: 0x4edd60
//
int ctor_244()
{
  int v0; // edx

  qword_4FB76C0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_4FB76D0 = 0;
  qword_4FB76D8 = 0;
  qword_4FB76E0 = 0;
  qword_4FB76E8 = 0;
  word_4FB76CC = word_4FB76CC & 0xF000 | 1;
  qword_4FB76F0 = 0;
  qword_4FB7708 = (__int64)qword_4FA01C0;
  qword_4FB7718 = (__int64)&unk_4FB7738;
  qword_4FB7720 = (__int64)&unk_4FB7738;
  dword_4FB76C8 = v0;
  qword_4FB76F8 = 0;
  qword_4FB76C0 = (__int64)&unk_49E75F8;
  qword_4FB7700 = 0;
  qword_4FB7710 = 0;
  qword_4FB7790 = (__int64)&unk_49EEE90;
  qword_4FB7728 = 4;
  dword_4FB7730 = 0;
  byte_4FB7758 = 0;
  qword_4FB7760 = 0;
  qword_4FB7768 = 0;
  qword_4FB7770 = 0;
  qword_4FB7778 = 0;
  qword_4FB7780 = 0;
  qword_4FB7788 = 0;
  sub_16B8280(&qword_4FB76C0, "rewrite-map-file", 16);
  qword_4FB76F0 = 18;
  qword_4FB76E8 = (__int64)"Symbol Rewrite Map";
  qword_4FB76F8 = (__int64)"filename";
  qword_4FB7700 = 8;
  LOBYTE(word_4FB76CC) = word_4FB76CC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FB76C0);
  return __cxa_atexit(sub_12F08D0, &qword_4FB76C0, &qword_4A427C0);
}

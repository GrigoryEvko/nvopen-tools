// Function: ctor_269
// Address: 0x4f65f0
//
int ctor_269()
{
  int v0; // edx

  qword_4FBE740 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FBE74C &= 0xF000u;
  qword_4FBE750 = 0;
  qword_4FBE788 = (__int64)qword_4FA01C0;
  qword_4FBE758 = 0;
  qword_4FBE760 = 0;
  qword_4FBE768 = 0;
  dword_4FBE748 = v0;
  qword_4FBE798 = (__int64)&unk_4FBE7B8;
  qword_4FBE7A0 = (__int64)&unk_4FBE7B8;
  qword_4FBE770 = 0;
  qword_4FBE778 = 0;
  qword_4FBE7E8 = (__int64)&unk_49E74E8;
  word_4FBE7F0 = 256;
  qword_4FBE780 = 0;
  qword_4FBE790 = 0;
  qword_4FBE740 = (__int64)&unk_49EEC70;
  qword_4FBE7A8 = 4;
  byte_4FBE7D8 = 0;
  qword_4FBE7F8 = (__int64)&unk_49EEDB0;
  dword_4FBE7B0 = 0;
  byte_4FBE7E0 = 0;
  sub_16B8280(&qword_4FBE740, "nvvm-lower-printf", 17);
  word_4FBE7F0 = 257;
  byte_4FBE7E0 = 1;
  qword_4FBE770 = 43;
  LOBYTE(word_4FBE74C) = word_4FBE74C & 0x9F | 0x20;
  qword_4FBE768 = (__int64)"Enable printf lowering (enabled by default)";
  sub_16B88A0(&qword_4FBE740);
  return __cxa_atexit(sub_12EDEC0, &qword_4FBE740, &qword_4A427C0);
}

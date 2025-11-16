// Function: ctor_212
// Address: 0x4e4620
//
int ctor_212()
{
  int v0; // edx

  qword_4FB0E40 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB0E4C &= 0xF000u;
  qword_4FB0E50 = 0;
  qword_4FB0E88 = (__int64)qword_4FA01C0;
  qword_4FB0E58 = 0;
  qword_4FB0E60 = 0;
  qword_4FB0E68 = 0;
  dword_4FB0E48 = v0;
  qword_4FB0E98 = (__int64)&unk_4FB0EB8;
  qword_4FB0EA0 = (__int64)&unk_4FB0EB8;
  qword_4FB0E70 = 0;
  qword_4FB0E78 = 0;
  qword_4FB0EE8 = (__int64)&unk_49E74A8;
  qword_4FB0E80 = 0;
  qword_4FB0E90 = 0;
  qword_4FB0E40 = (__int64)&unk_49EEAF0;
  qword_4FB0EA8 = 4;
  dword_4FB0EB0 = 0;
  qword_4FB0EF8 = (__int64)&unk_49EEE10;
  byte_4FB0ED8 = 0;
  dword_4FB0EE0 = 0;
  byte_4FB0EF4 = 1;
  dword_4FB0EF0 = 0;
  sub_16B8280(&qword_4FB0E40, "rotation-max-header-size", 24);
  dword_4FB0EE0 = 16;
  byte_4FB0EF4 = 1;
  dword_4FB0EF0 = 16;
  qword_4FB0E70 = 59;
  LOBYTE(word_4FB0E4C) = word_4FB0E4C & 0x9F | 0x20;
  qword_4FB0E68 = (__int64)"The default maximum header size for automatic loop rotation";
  sub_16B88A0(&qword_4FB0E40);
  return __cxa_atexit(sub_12EDE60, &qword_4FB0E40, &qword_4A427C0);
}

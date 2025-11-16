// Function: ctor_273
// Address: 0x4f7250
//
int ctor_273()
{
  int v0; // eax
  int v1; // eax

  qword_4FBEF20 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_4FBEF30 = 0;
  qword_4FBEF68 = (__int64)qword_4FA01C0;
  qword_4FBEF38 = 0;
  qword_4FBEF40 = 0;
  qword_4FBEF48 = 0;
  dword_4FBEF28 = v0;
  qword_4FBEF50 = 0;
  qword_4FBEF58 = 0;
  qword_4FBEF60 = 0;
  word_4FBEF2C = word_4FBEF2C & 0xF000 | 1;
  qword_4FBEF78 = (__int64)&unk_4FBEF98;
  qword_4FBEF80 = (__int64)&unk_4FBEF98;
  qword_4FBEF70 = 0;
  byte_4FBEFB8 = 0;
  qword_4FBEF20 = (__int64)&unk_49E75F8;
  qword_4FBEF88 = 4;
  dword_4FBEF90 = 0;
  qword_4FBEFF0 = (__int64)&unk_49EEE90;
  qword_4FBEFC0 = 0;
  qword_4FBEFC8 = 0;
  qword_4FBEFD0 = 0;
  qword_4FBEFD8 = 0;
  qword_4FBEFE0 = 0;
  qword_4FBEFE8 = 0;
  sub_16B8280(&qword_4FBEF20, "select-kernel-list", 18);
  HIBYTE(word_4FBEF2C) |= 2u;
  qword_4FBEF58 = (__int64)"list";
  qword_4FBEF48 = (__int64)"A list of kernel to optimize";
  qword_4FBEF60 = 4;
  qword_4FBEF50 = 28;
  sub_16B88A0(&qword_4FBEF20);
  __cxa_atexit(sub_12F08D0, &qword_4FBEF20, &qword_4A427C0);
  qword_4FBEE40 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_4FBEE50 = 0;
  qword_4FBEE58 = 0;
  qword_4FBEE60 = 0;
  qword_4FBEE68 = 0;
  qword_4FBEE70 = 0;
  qword_4FBEE78 = 0;
  dword_4FBEE48 = v1;
  qword_4FBEE88 = (__int64)qword_4FA01C0;
  qword_4FBEE80 = 0;
  qword_4FBEE90 = 0;
  word_4FBEE4C = word_4FBEE4C & 0xF000 | 1;
  qword_4FBEE98 = (__int64)&unk_4FBEEB8;
  qword_4FBEEA0 = (__int64)&unk_4FBEEB8;
  qword_4FBEEA8 = 4;
  dword_4FBEEB0 = 0;
  qword_4FBEE40 = (__int64)&unk_49F88A0;
  byte_4FBEED8 = 0;
  qword_4FBEEE0 = 0;
  qword_4FBEF10 = (__int64)&unk_49EEE10;
  qword_4FBEEE8 = 0;
  qword_4FBEEF0 = 0;
  qword_4FBEEF8 = 0;
  qword_4FBEF00 = 0;
  qword_4FBEF08 = 0;
  sub_16B8280(&qword_4FBEE40, "select-kernel-range", 19);
  HIBYTE(word_4FBEE4C) |= 2u;
  qword_4FBEE78 = (__int64)"list";
  qword_4FBEE80 = 4;
  qword_4FBEE68 = (__int64)"A set of kernels to optimize";
  qword_4FBEE70 = 28;
  sub_16B88A0(&qword_4FBEE40);
  return __cxa_atexit(sub_1CC3420, &qword_4FBEE40, &qword_4A427C0);
}

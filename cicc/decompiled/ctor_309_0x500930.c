// Function: ctor_309
// Address: 0x500930
//
int ctor_309()
{
  int v0; // edx

  qword_4FC7200 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC720C &= 0xF000u;
  qword_4FC7210 = 0;
  qword_4FC7248 = (__int64)qword_4FA01C0;
  qword_4FC7218 = 0;
  qword_4FC7220 = 0;
  qword_4FC7228 = 0;
  dword_4FC7208 = v0;
  qword_4FC7258 = (__int64)&unk_4FC7278;
  qword_4FC7260 = (__int64)&unk_4FC7278;
  qword_4FC7230 = 0;
  qword_4FC7238 = 0;
  qword_4FC72A8 = (__int64)&unk_49E74E8;
  word_4FC72B0 = 256;
  qword_4FC7240 = 0;
  qword_4FC7250 = 0;
  qword_4FC7200 = (__int64)&unk_49EEC70;
  qword_4FC7268 = 4;
  byte_4FC7298 = 0;
  qword_4FC72B8 = (__int64)&unk_49EEDB0;
  dword_4FC7270 = 0;
  byte_4FC72A0 = 0;
  sub_16B8280(&qword_4FC7200, "enable-subreg-liveness", 22);
  word_4FC72B0 = 257;
  byte_4FC72A0 = 1;
  qword_4FC7230 = 37;
  LOBYTE(word_4FC720C) = word_4FC720C & 0x9F | 0x20;
  qword_4FC7228 = (__int64)"Enable subregister liveness tracking.";
  sub_16B88A0(&qword_4FC7200);
  return __cxa_atexit(sub_12EDEC0, &qword_4FC7200, &qword_4A427C0);
}

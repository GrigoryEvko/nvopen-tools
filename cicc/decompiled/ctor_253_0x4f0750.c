// Function: ctor_253
// Address: 0x4f0750
//
int ctor_253()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax

  qword_4FBA2A0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FBA2AC &= 0xF000u;
  qword_4FBA2B0 = 0;
  qword_4FBA2B8 = 0;
  qword_4FBA2C0 = 0;
  qword_4FBA2C8 = 0;
  qword_4FBA2D0 = 0;
  dword_4FBA2A8 = v0;
  qword_4FBA2D8 = 0;
  qword_4FBA2E8 = (__int64)qword_4FA01C0;
  qword_4FBA2F8 = (__int64)&unk_4FBA318;
  qword_4FBA300 = (__int64)&unk_4FBA318;
  qword_4FBA2E0 = 0;
  qword_4FBA2F0 = 0;
  word_4FBA350 = 256;
  qword_4FBA348 = (__int64)&unk_49E74E8;
  qword_4FBA2A0 = (__int64)&unk_49EEC70;
  qword_4FBA358 = (__int64)&unk_49EEDB0;
  qword_4FBA308 = 4;
  dword_4FBA310 = 0;
  byte_4FBA338 = 0;
  byte_4FBA340 = 0;
  sub_16B8280(&qword_4FBA2A0, "dump-va", 7);
  word_4FBA350 = 256;
  byte_4FBA340 = 0;
  qword_4FBA2D0 = 33;
  LOBYTE(word_4FBA2AC) = word_4FBA2AC & 0x9F | 0x20;
  qword_4FBA2C8 = (__int64)"Dump result from variance inquiry";
  sub_16B88A0(&qword_4FBA2A0);
  __cxa_atexit(sub_12EDEC0, &qword_4FBA2A0, &qword_4A427C0);
  qword_4FBA1C0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FBA1CC &= 0xF000u;
  qword_4FBA1D0 = 0;
  qword_4FBA1D8 = 0;
  qword_4FBA1E0 = 0;
  qword_4FBA1E8 = 0;
  qword_4FBA1F0 = 0;
  dword_4FBA1C8 = v1;
  qword_4FBA1F8 = 0;
  qword_4FBA208 = (__int64)qword_4FA01C0;
  qword_4FBA218 = (__int64)&unk_4FBA238;
  qword_4FBA220 = (__int64)&unk_4FBA238;
  qword_4FBA200 = 0;
  qword_4FBA210 = 0;
  qword_4FBA268 = (__int64)&unk_49E74A8;
  qword_4FBA228 = 4;
  dword_4FBA230 = 0;
  qword_4FBA1C0 = (__int64)&unk_49EEAF0;
  byte_4FBA258 = 0;
  dword_4FBA260 = 0;
  qword_4FBA278 = (__int64)&unk_49EEE10;
  byte_4FBA274 = 1;
  dword_4FBA270 = 0;
  sub_16B8280(&qword_4FBA1C0, "variance-analysis-limit", 23);
  dword_4FBA260 = 10000;
  byte_4FBA274 = 1;
  dword_4FBA270 = 10000;
  qword_4FBA1F0 = 47;
  LOBYTE(word_4FBA1CC) = word_4FBA1CC & 0x9F | 0x20;
  qword_4FBA1E8 = (__int64)"Control the function size for variance analysis";
  sub_16B88A0(&qword_4FBA1C0);
  __cxa_atexit(sub_12EDE60, &qword_4FBA1C0, &qword_4A427C0);
  qword_4FBA0E0 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FBA190 = 256;
  qword_4FBA0F0 = 0;
  word_4FBA0EC &= 0xF000u;
  qword_4FBA188 = (__int64)&unk_49E74E8;
  qword_4FBA0E0 = (__int64)&unk_49EEC70;
  dword_4FBA0E8 = v2;
  qword_4FBA198 = (__int64)&unk_49EEDB0;
  qword_4FBA128 = (__int64)qword_4FA01C0;
  qword_4FBA138 = (__int64)&unk_4FBA158;
  qword_4FBA140 = (__int64)&unk_4FBA158;
  qword_4FBA0F8 = 0;
  qword_4FBA100 = 0;
  qword_4FBA108 = 0;
  qword_4FBA110 = 0;
  qword_4FBA118 = 0;
  qword_4FBA120 = 0;
  qword_4FBA130 = 0;
  qword_4FBA148 = 4;
  dword_4FBA150 = 0;
  byte_4FBA178 = 0;
  byte_4FBA180 = 0;
  sub_16B8280(&qword_4FBA0E0, "va-use-scdg", 11);
  word_4FBA190 = 257;
  byte_4FBA180 = 1;
  qword_4FBA110 = 72;
  LOBYTE(word_4FBA0EC) = word_4FBA0EC & 0x9F | 0x20;
  qword_4FBA108 = (__int64)"Control if the properties of structured control dependence graph is used";
  sub_16B88A0(&qword_4FBA0E0);
  return __cxa_atexit(sub_12EDEC0, &qword_4FBA0E0, &qword_4A427C0);
}

// Function: ctor_315
// Address: 0x502c30
//
int ctor_315()
{
  int v0; // edx
  char v2; // [rsp+13h] [rbp-4Dh] BYREF
  int v3; // [rsp+14h] [rbp-4Ch] BYREF
  char *v4; // [rsp+18h] [rbp-48h] BYREF
  const char *v5; // [rsp+20h] [rbp-40h]
  __int64 v6; // [rsp+28h] [rbp-38h]

  v5 = "Disable critical edge splitting during PHI elimination";
  v6 = 54;
  v3 = 1;
  v2 = 0;
  v4 = &v2;
  sub_199AE00(&unk_4FC8BE0, "disable-phi-elim-edge-splitting", &v4, &v3);
  __cxa_atexit(sub_12EDEC0, &unk_4FC8BE0, &qword_4A427C0);
  qword_4FC8B00 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC8B0C &= 0xF000u;
  qword_4FC8B10 = 0;
  qword_4FC8B48 = (__int64)qword_4FA01C0;
  qword_4FC8B18 = 0;
  qword_4FC8B20 = 0;
  qword_4FC8B28 = 0;
  dword_4FC8B08 = v0;
  qword_4FC8B58 = (__int64)&unk_4FC8B78;
  qword_4FC8B60 = (__int64)&unk_4FC8B78;
  qword_4FC8B30 = 0;
  qword_4FC8B38 = 0;
  qword_4FC8BA8 = (__int64)&unk_49E74E8;
  word_4FC8BB0 = 256;
  qword_4FC8B40 = 0;
  qword_4FC8B50 = 0;
  qword_4FC8B00 = (__int64)&unk_49EEC70;
  qword_4FC8B68 = 4;
  byte_4FC8B98 = 0;
  qword_4FC8BB8 = (__int64)&unk_49EEDB0;
  dword_4FC8B70 = 0;
  byte_4FC8BA0 = 0;
  sub_16B8280(&qword_4FC8B00, "phi-elim-split-all-critical-edges", 33);
  word_4FC8BB0 = 256;
  byte_4FC8BA0 = 0;
  qword_4FC8B30 = 47;
  LOBYTE(word_4FC8B0C) = word_4FC8B0C & 0x9F | 0x20;
  qword_4FC8B28 = (__int64)"Split all critical edges during PHI elimination";
  sub_16B88A0(&qword_4FC8B00);
  __cxa_atexit(sub_12EDEC0, &qword_4FC8B00, &qword_4A427C0);
  v4 = &v2;
  v5 = "Do not use an early exit if isLiveOutPastPHIs returns true.";
  v6 = 59;
  v3 = 1;
  v2 = 0;
  sub_199AE00(&unk_4FC8A20, "no-phi-elim-live-out-early-exit", &v4, &v3);
  return __cxa_atexit(sub_12EDEC0, &unk_4FC8A20, &qword_4A427C0);
}

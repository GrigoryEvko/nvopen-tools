// Function: ctor_230
// Address: 0x4ea750
//
int ctor_230()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax
  char v4; // [rsp+23h] [rbp-4Dh] BYREF
  int v5; // [rsp+24h] [rbp-4Ch] BYREF
  char *v6; // [rsp+28h] [rbp-48h] BYREF
  const char *v7; // [rsp+30h] [rbp-40h] BYREF
  __int64 v8; // [rsp+38h] [rbp-38h]

  v6 = &v4;
  v7 = "Turn on DominatorTree and LoopInfo verification after Loop Distribution";
  v4 = 0;
  v8 = 71;
  v5 = 1;
  sub_1A8B390(&unk_4FB5640, "loop-distribute-verify", &v5, &v7, &v6);
  __cxa_atexit(sub_12EDEC0, &unk_4FB5640, &qword_4A427C0);
  qword_4FB5560 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB556C &= 0xF000u;
  qword_4FB5570 = 0;
  qword_4FB5578 = 0;
  qword_4FB5580 = 0;
  qword_4FB5588 = 0;
  qword_4FB5590 = 0;
  dword_4FB5568 = v0;
  qword_4FB5598 = 0;
  qword_4FB55A8 = (__int64)qword_4FA01C0;
  qword_4FB55B8 = (__int64)&unk_4FB55D8;
  qword_4FB55C0 = (__int64)&unk_4FB55D8;
  qword_4FB55A0 = 0;
  qword_4FB55B0 = 0;
  qword_4FB5608 = (__int64)&unk_49E74E8;
  word_4FB5610 = 256;
  qword_4FB55C8 = 4;
  byte_4FB55F8 = 0;
  qword_4FB5560 = (__int64)&unk_49EEC70;
  dword_4FB55D0 = 0;
  byte_4FB5600 = 0;
  qword_4FB5618 = (__int64)&unk_49EEDB0;
  sub_16B8280(&qword_4FB5560, "loop-distribute-non-if-convertible", 34);
  word_4FB5610 = 256;
  byte_4FB5600 = 0;
  qword_4FB5590 = 87;
  LOBYTE(word_4FB556C) = word_4FB556C & 0x9F | 0x20;
  qword_4FB5588 = (__int64)"Whether to distribute into a loop that may not be if-convertible by the loop vectorizer";
  sub_16B88A0(&qword_4FB5560);
  __cxa_atexit(sub_12EDEC0, &qword_4FB5560, &qword_4A427C0);
  qword_4FB5480 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB548C &= 0xF000u;
  qword_4FB5490 = 0;
  qword_4FB5498 = 0;
  qword_4FB54A0 = 0;
  qword_4FB54A8 = 0;
  qword_4FB54B0 = 0;
  dword_4FB5488 = v1;
  qword_4FB54B8 = 0;
  qword_4FB54C8 = (__int64)qword_4FA01C0;
  qword_4FB54D8 = (__int64)&unk_4FB54F8;
  qword_4FB54E0 = (__int64)&unk_4FB54F8;
  qword_4FB54C0 = 0;
  qword_4FB54D0 = 0;
  qword_4FB5528 = (__int64)&unk_49E74A8;
  qword_4FB5480 = (__int64)&unk_49EEAF0;
  qword_4FB5538 = (__int64)&unk_49EEE10;
  qword_4FB54E8 = 4;
  dword_4FB54F0 = 0;
  byte_4FB5518 = 0;
  dword_4FB5520 = 0;
  byte_4FB5534 = 1;
  dword_4FB5530 = 0;
  sub_16B8280(&qword_4FB5480, "loop-distribute-scev-check-threshold", 36);
  dword_4FB5520 = 8;
  byte_4FB5534 = 1;
  dword_4FB5530 = 8;
  qword_4FB54B0 = 63;
  LOBYTE(word_4FB548C) = word_4FB548C & 0x9F | 0x20;
  qword_4FB54A8 = (__int64)"The maximum number of SCEV checks allowed for Loop Distribution";
  sub_16B88A0(&qword_4FB5480);
  __cxa_atexit(sub_12EDE60, &qword_4FB5480, &qword_4A427C0);
  qword_4FB53A0 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB53AC &= 0xF000u;
  qword_4FB53B0 = 0;
  qword_4FB53B8 = 0;
  qword_4FB53C0 = 0;
  qword_4FB5448 = (__int64)&unk_49E74A8;
  qword_4FB53A0 = (__int64)&unk_49EEAF0;
  dword_4FB53A8 = v2;
  qword_4FB5458 = (__int64)&unk_49EEE10;
  qword_4FB53E8 = (__int64)qword_4FA01C0;
  qword_4FB53F8 = (__int64)&unk_4FB5418;
  qword_4FB5400 = (__int64)&unk_4FB5418;
  qword_4FB53C8 = 0;
  qword_4FB53D0 = 0;
  qword_4FB53D8 = 0;
  qword_4FB53E0 = 0;
  qword_4FB53F0 = 0;
  qword_4FB5408 = 4;
  dword_4FB5410 = 0;
  byte_4FB5438 = 0;
  dword_4FB5440 = 0;
  byte_4FB5454 = 1;
  dword_4FB5450 = 0;
  sub_16B8280(&qword_4FB53A0, "loop-distribute-scev-check-threshold-with-pragma", 48);
  dword_4FB5440 = 128;
  byte_4FB5454 = 1;
  dword_4FB5450 = 128;
  qword_4FB53D0 = 116;
  LOBYTE(word_4FB53AC) = word_4FB53AC & 0x9F | 0x20;
  qword_4FB53C8 = (__int64)"The maximum number of SCEV checks allowed for Loop Distribution for loop marked with #pragma "
                           "loop distribute(enable)";
  sub_16B88A0(&qword_4FB53A0);
  __cxa_atexit(sub_12EDE60, &qword_4FB53A0, &qword_4A427C0);
  v4 = 0;
  v6 = &v4;
  v7 = "Enable the new, experimental LoopDistribution Pass";
  v8 = 50;
  v5 = 1;
  sub_1A8B390(&unk_4FB52C0, "enable-loop-distribute", &v5, &v7, &v6);
  return __cxa_atexit(sub_12EDEC0, &unk_4FB52C0, &qword_4A427C0);
}

// Function: ctor_206
// Address: 0x4e33a0
//
int ctor_206()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax

  qword_4FB0580 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB058C &= 0xF000u;
  qword_4FB05C8 = (__int64)qword_4FA01C0;
  qword_4FB0590 = 0;
  qword_4FB0598 = 0;
  qword_4FB05A0 = 0;
  dword_4FB0588 = v0;
  qword_4FB05D8 = (__int64)&unk_4FB05F8;
  qword_4FB05E0 = (__int64)&unk_4FB05F8;
  qword_4FB05A8 = 0;
  qword_4FB05B0 = 0;
  qword_4FB0628 = (__int64)&unk_49E74E8;
  word_4FB0630 = 256;
  qword_4FB05B8 = 0;
  qword_4FB05C0 = 0;
  qword_4FB0580 = (__int64)&unk_49EEC70;
  qword_4FB05D0 = 0;
  byte_4FB0618 = 0;
  qword_4FB0638 = (__int64)&unk_49EEDB0;
  qword_4FB05E8 = 4;
  dword_4FB05F0 = 0;
  byte_4FB0620 = 0;
  sub_16B8280(&qword_4FB0580, "disable-licm-promotion", 22);
  word_4FB0630 = 256;
  byte_4FB0620 = 0;
  qword_4FB05B0 = 37;
  LOBYTE(word_4FB058C) = word_4FB058C & 0x9F | 0x20;
  qword_4FB05A8 = (__int64)"Disable memory promotion in LICM pass";
  sub_16B88A0(&qword_4FB0580);
  __cxa_atexit(sub_12EDEC0, &qword_4FB0580, &qword_4A427C0);
  qword_4FB04A0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB04AC &= 0xF000u;
  qword_4FB04B0 = 0;
  qword_4FB04B8 = 0;
  qword_4FB04C0 = 0;
  qword_4FB04C8 = 0;
  qword_4FB04D0 = 0;
  dword_4FB04A8 = v1;
  qword_4FB04F8 = (__int64)&unk_4FB0518;
  qword_4FB0500 = (__int64)&unk_4FB0518;
  qword_4FB04E8 = (__int64)qword_4FA01C0;
  qword_4FB04D8 = 0;
  qword_4FB0548 = (__int64)&unk_49E74A8;
  qword_4FB04A0 = (__int64)&unk_49EEAF0;
  qword_4FB0558 = (__int64)&unk_49EEE10;
  qword_4FB04E0 = 0;
  qword_4FB04F0 = 0;
  qword_4FB0508 = 4;
  dword_4FB0510 = 0;
  byte_4FB0538 = 0;
  dword_4FB0540 = 0;
  byte_4FB0554 = 1;
  dword_4FB0550 = 0;
  sub_16B8280(&qword_4FB04A0, "licm-max-num-uses-traversed", 27);
  dword_4FB0540 = 8;
  byte_4FB0554 = 1;
  dword_4FB0550 = 8;
  qword_4FB04D0 = 96;
  LOBYTE(word_4FB04AC) = word_4FB04AC & 0x9F | 0x20;
  qword_4FB04C8 = (__int64)"Max num uses visited for identifying load invariance in loop using invariant start (default = 8)";
  sub_16B88A0(&qword_4FB04A0);
  __cxa_atexit(sub_12EDE60, &qword_4FB04A0, &qword_4A427C0);
  qword_4FB03C0 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB03CC &= 0xF000u;
  qword_4FB03D0 = 0;
  qword_4FB03D8 = 0;
  qword_4FB03E0 = 0;
  qword_4FB0468 = (__int64)&unk_49E74A8;
  qword_4FB03C0 = (__int64)&unk_49EEAF0;
  dword_4FB03C8 = v2;
  qword_4FB0408 = (__int64)qword_4FA01C0;
  qword_4FB0418 = (__int64)&unk_4FB0438;
  qword_4FB0420 = (__int64)&unk_4FB0438;
  qword_4FB0478 = (__int64)&unk_49EEE10;
  qword_4FB03E8 = 0;
  qword_4FB03F0 = 0;
  qword_4FB03F8 = 0;
  qword_4FB0400 = 0;
  qword_4FB0410 = 0;
  qword_4FB0428 = 4;
  dword_4FB0430 = 0;
  byte_4FB0458 = 0;
  dword_4FB0460 = 0;
  byte_4FB0474 = 1;
  dword_4FB0470 = 0;
  sub_16B8280(&qword_4FB03C0, "licm-insn-limit", 15);
  dword_4FB0460 = 500;
  byte_4FB0474 = 1;
  dword_4FB0470 = 500;
  qword_4FB03F0 = 40;
  LOBYTE(word_4FB03CC) = word_4FB03CC & 0x9F | 0x20;
  qword_4FB03E8 = (__int64)"Control the loop-size threshold for LICM";
  sub_16B88A0(&qword_4FB03C0);
  return __cxa_atexit(sub_12EDE60, &qword_4FB03C0, &qword_4A427C0);
}

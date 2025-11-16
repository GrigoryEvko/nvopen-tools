// Function: ctor_167
// Address: 0x4d1690
//
int ctor_167()
{
  int v0; // eax
  int v1; // eax

  qword_4FA2720 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA272C &= 0xF000u;
  qword_4FA2768 = (__int64)qword_4FA01C0;
  qword_4FA2730 = 0;
  qword_4FA2738 = 0;
  qword_4FA2740 = 0;
  dword_4FA2728 = v0;
  qword_4FA2778 = (__int64)&unk_4FA2798;
  qword_4FA2780 = (__int64)&unk_4FA2798;
  qword_4FA2748 = 0;
  qword_4FA2750 = 0;
  qword_4FA27C8 = (__int64)&unk_49E74A8;
  qword_4FA2758 = 0;
  qword_4FA2760 = 0;
  qword_4FA2720 = (__int64)&unk_49EEAF0;
  qword_4FA2770 = 0;
  byte_4FA27B8 = 0;
  qword_4FA27D8 = (__int64)&unk_49EEE10;
  qword_4FA2788 = 4;
  dword_4FA2790 = 0;
  dword_4FA27C0 = 0;
  byte_4FA27D4 = 1;
  dword_4FA27D0 = 0;
  sub_16B8280(&qword_4FA2720, "max-aggr-lower-size", 19);
  dword_4FA27C0 = 128;
  byte_4FA27D4 = 1;
  dword_4FA27D0 = 128;
  qword_4FA2750 = 60;
  LOBYTE(word_4FA272C) = word_4FA272C & 0x9F | 0x20;
  qword_4FA2748 = (__int64)"The threshold size below which its okay to lower aggregates.";
  sub_16B88A0(&qword_4FA2720);
  __cxa_atexit(sub_12EDE60, &qword_4FA2720, &qword_4A427C0);
  qword_4FA2640 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA264C &= 0xF000u;
  qword_4FA2650 = 0;
  qword_4FA2658 = 0;
  qword_4FA2660 = 0;
  qword_4FA2668 = 0;
  qword_4FA2670 = 0;
  dword_4FA2648 = v1;
  qword_4FA2698 = (__int64)&unk_4FA26B8;
  qword_4FA26A0 = (__int64)&unk_4FA26B8;
  qword_4FA2688 = (__int64)qword_4FA01C0;
  qword_4FA2678 = 0;
  qword_4FA26E8 = (__int64)&unk_49E74E8;
  word_4FA26F0 = 256;
  qword_4FA2680 = 0;
  qword_4FA2690 = 0;
  qword_4FA2640 = (__int64)&unk_49EEC70;
  qword_4FA26A8 = 4;
  byte_4FA26D8 = 0;
  qword_4FA26F8 = (__int64)&unk_49EEDB0;
  dword_4FA26B0 = 0;
  byte_4FA26E0 = 0;
  sub_16B8280(&qword_4FA2640, "disable-load-select-transform", 29);
  word_4FA26F0 = 256;
  byte_4FA26E0 = 0;
  qword_4FA2670 = 58;
  LOBYTE(word_4FA264C) = word_4FA264C & 0x9F | 0x20;
  qword_4FA2668 = (__int64)"Disable ld(sel a1, a2) -> sel(ld v1, ld v2) transformation";
  sub_16B88A0(&qword_4FA2640);
  return __cxa_atexit(sub_12EDEC0, &qword_4FA2640, &qword_4A427C0);
}

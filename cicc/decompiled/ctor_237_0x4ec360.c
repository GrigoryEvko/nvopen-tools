// Function: ctor_237
// Address: 0x4ec360
//
int ctor_237()
{
  int v0; // eax
  int v1; // eax

  qword_4FB67C0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB67CC &= 0xF000u;
  qword_4FB67D0 = 0;
  qword_4FB67D8 = 0;
  qword_4FB67E0 = 0;
  qword_4FB67E8 = 0;
  qword_4FB67F0 = 0;
  dword_4FB67C8 = v0;
  qword_4FB67F8 = 0;
  qword_4FB6808 = (__int64)qword_4FA01C0;
  qword_4FB6818 = (__int64)&unk_4FB6838;
  qword_4FB6820 = (__int64)&unk_4FB6838;
  qword_4FB6800 = 0;
  qword_4FB6810 = 0;
  word_4FB6870 = 256;
  qword_4FB6868 = (__int64)&unk_49E74E8;
  qword_4FB6828 = 4;
  qword_4FB67C0 = (__int64)&unk_49EEC70;
  byte_4FB6858 = 0;
  qword_4FB6878 = (__int64)&unk_49EEDB0;
  dword_4FB6830 = 0;
  byte_4FB6860 = 0;
  sub_16B8280(&qword_4FB67C0, "unroll-runtime-epilog", 21);
  word_4FB6870 = 256;
  byte_4FB6860 = 0;
  qword_4FB67F0 = 74;
  LOBYTE(word_4FB67CC) = word_4FB67CC & 0x9F | 0x20;
  qword_4FB67E8 = (__int64)"Allow runtime unrolled loops to be unrolled with epilog instead of prolog.";
  sub_16B88A0(&qword_4FB67C0);
  __cxa_atexit(sub_12EDEC0, &qword_4FB67C0, &qword_4A427C0);
  qword_4FB66E0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB6790 = 256;
  word_4FB66EC &= 0xF000u;
  qword_4FB66F0 = 0;
  qword_4FB66F8 = 0;
  qword_4FB6700 = 0;
  dword_4FB66E8 = v1;
  qword_4FB6788 = (__int64)&unk_49E74E8;
  qword_4FB6728 = (__int64)qword_4FA01C0;
  qword_4FB6738 = (__int64)&unk_4FB6758;
  qword_4FB6740 = (__int64)&unk_4FB6758;
  qword_4FB66E0 = (__int64)&unk_49EEC70;
  qword_4FB6798 = (__int64)&unk_49EEDB0;
  qword_4FB6708 = 0;
  qword_4FB6710 = 0;
  qword_4FB6718 = 0;
  qword_4FB6720 = 0;
  qword_4FB6730 = 0;
  qword_4FB6748 = 4;
  dword_4FB6750 = 0;
  byte_4FB6778 = 0;
  byte_4FB6780 = 0;
  sub_16B8280(&qword_4FB66E0, "unroll-verify-domtree", 21);
  word_4FB6790 = 256;
  byte_4FB6780 = 0;
  qword_4FB6710 = 30;
  LOBYTE(word_4FB66EC) = word_4FB66EC & 0x9F | 0x20;
  qword_4FB6708 = (__int64)"Verify domtree after unrolling";
  sub_16B88A0(&qword_4FB66E0);
  return __cxa_atexit(sub_12EDEC0, &qword_4FB66E0, &qword_4A427C0);
}

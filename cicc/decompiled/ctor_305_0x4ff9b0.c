// Function: ctor_305
// Address: 0x4ff9b0
//
int ctor_305()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax
  int v3; // eax
  char v5; // [rsp+23h] [rbp-4Dh] BYREF
  int v6; // [rsp+24h] [rbp-4Ch] BYREF
  char *v7; // [rsp+28h] [rbp-48h] BYREF
  const char *v8; // [rsp+30h] [rbp-40h] BYREF
  __int64 v9; // [rsp+38h] [rbp-38h]

  v7 = &v5;
  v8 = "MachineLICM should avoid speculation";
  v6 = 1;
  v5 = 1;
  v9 = 36;
  sub_1E1E0A0(&unk_4FC6940, "avoid-speculation", &v8, &v7, &v6);
  __cxa_atexit(sub_12EDEC0, &unk_4FC6940, &qword_4A427C0);
  v7 = &v5;
  v8 = "MachineLICM should hoist even cheap instructions";
  v6 = 1;
  v5 = 0;
  v9 = 48;
  sub_1E1E0A0(&unk_4FC6860, "hoist-cheap-insts", &v8, &v7, &v6);
  __cxa_atexit(sub_12EDEC0, &unk_4FC6860, &qword_4A427C0);
  qword_4FC6780 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC678C &= 0xF000u;
  qword_4FC67C8 = (__int64)qword_4FA01C0;
  qword_4FC6790 = 0;
  qword_4FC6798 = 0;
  qword_4FC67A0 = 0;
  dword_4FC6788 = v0;
  qword_4FC67D8 = (__int64)&unk_4FC67F8;
  qword_4FC67E0 = (__int64)&unk_4FC67F8;
  qword_4FC67A8 = 0;
  qword_4FC67B0 = 0;
  word_4FC6830 = 256;
  qword_4FC6828 = (__int64)&unk_49E74E8;
  qword_4FC6780 = (__int64)&unk_49EEC70;
  qword_4FC6838 = (__int64)&unk_49EEDB0;
  qword_4FC67B8 = 0;
  qword_4FC67C0 = 0;
  qword_4FC67D0 = 0;
  qword_4FC67E8 = 4;
  dword_4FC67F0 = 0;
  byte_4FC6818 = 0;
  byte_4FC6820 = 0;
  sub_16B8280(&qword_4FC6780, "sink-insts-to-avoid-spills", 26);
  qword_4FC67A8 = (__int64)"MachineLICM should sink instructions into loops to avoid register spills";
  word_4FC6830 = 256;
  byte_4FC6820 = 0;
  qword_4FC67B0 = 72;
  LOBYTE(word_4FC678C) = word_4FC678C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC6780);
  __cxa_atexit(sub_12EDEC0, &qword_4FC6780, &qword_4A427C0);
  qword_4FC66A0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC6750 = 256;
  qword_4FC66B0 = 0;
  word_4FC66AC &= 0xF000u;
  qword_4FC6748 = (__int64)&unk_49E74E8;
  qword_4FC66A0 = (__int64)&unk_49EEC70;
  dword_4FC66A8 = v1;
  qword_4FC6758 = (__int64)&unk_49EEDB0;
  qword_4FC66E8 = (__int64)qword_4FA01C0;
  qword_4FC66F8 = (__int64)&unk_4FC6718;
  qword_4FC6700 = (__int64)&unk_4FC6718;
  qword_4FC66B8 = 0;
  qword_4FC66C0 = 0;
  qword_4FC66C8 = 0;
  qword_4FC66D0 = 0;
  qword_4FC66D8 = 0;
  qword_4FC66E0 = 0;
  qword_4FC66F0 = 0;
  qword_4FC6708 = 4;
  dword_4FC6710 = 0;
  byte_4FC6738 = 0;
  byte_4FC6740 = 0;
  sub_16B8280(&qword_4FC66A0, "hoist-const-stores", 18);
  qword_4FC66C8 = (__int64)"Hoist invariant stores";
  word_4FC6750 = 257;
  byte_4FC6740 = 1;
  qword_4FC66D0 = 22;
  LOBYTE(word_4FC66AC) = word_4FC66AC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC66A0);
  __cxa_atexit(sub_12EDEC0, &qword_4FC66A0, &qword_4A427C0);
  qword_4FC65C0 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC65CC &= 0xF000u;
  qword_4FC65D0 = 0;
  qword_4FC65D8 = 0;
  qword_4FC65E0 = 0;
  qword_4FC65E8 = 0;
  qword_4FC65F0 = 0;
  dword_4FC65C8 = v2;
  qword_4FC6618 = (__int64)&unk_4FC6638;
  qword_4FC6620 = (__int64)&unk_4FC6638;
  qword_4FC6608 = (__int64)qword_4FA01C0;
  qword_4FC65F8 = 0;
  qword_4FC6668 = (__int64)&unk_49E74C8;
  qword_4FC65C0 = (__int64)&unk_49EEB70;
  qword_4FC6678 = (__int64)&unk_49EEDF0;
  qword_4FC6600 = 0;
  qword_4FC6610 = 0;
  qword_4FC6628 = 4;
  dword_4FC6630 = 0;
  byte_4FC6658 = 0;
  dword_4FC6660 = 0;
  byte_4FC6674 = 1;
  dword_4FC6670 = 0;
  sub_16B8280(&qword_4FC65C0, "heavy-const-expr-size", 21);
  qword_4FC65F0 = 48;
  qword_4FC65E8 = (__int64)"Size of heavy const exprs that should be hoisted";
  dword_4FC6660 = 7;
  byte_4FC6674 = 1;
  dword_4FC6670 = 7;
  LOBYTE(word_4FC65CC) = word_4FC65CC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC65C0);
  __cxa_atexit(sub_12EDEA0, &qword_4FC65C0, &qword_4A427C0);
  qword_4FC64E0 = (__int64)&unk_49EED30;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC64EC &= 0xF000u;
  qword_4FC64F0 = 0;
  qword_4FC64F8 = 0;
  qword_4FC6500 = 0;
  qword_4FC6588 = (__int64)&unk_49E74C8;
  qword_4FC64E0 = (__int64)&unk_49EEB70;
  dword_4FC64E8 = v3;
  qword_4FC6528 = (__int64)qword_4FA01C0;
  qword_4FC6538 = (__int64)&unk_4FC6558;
  qword_4FC6540 = (__int64)&unk_4FC6558;
  qword_4FC6598 = (__int64)&unk_49EEDF0;
  qword_4FC6508 = 0;
  qword_4FC6510 = 0;
  qword_4FC6518 = 0;
  qword_4FC6520 = 0;
  qword_4FC6530 = 0;
  qword_4FC6548 = 4;
  dword_4FC6550 = 0;
  byte_4FC6578 = 0;
  dword_4FC6580 = 0;
  byte_4FC6594 = 1;
  dword_4FC6590 = 0;
  sub_16B8280(&qword_4FC64E0, "const-expr-max-use", 18);
  qword_4FC6510 = 50;
  qword_4FC6508 = (__int64)"Stop hoisting const-exprs if it has too many users";
  dword_4FC6580 = 3;
  byte_4FC6594 = 1;
  dword_4FC6590 = 3;
  LOBYTE(word_4FC64EC) = word_4FC64EC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC64E0);
  return __cxa_atexit(sub_12EDEA0, &qword_4FC64E0, &qword_4A427C0);
}

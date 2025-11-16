// Function: ctor_280
// Address: 0x4f89c0
//
__int64 ctor_280()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 result; // rax
  _QWORD v6[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v7[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v8[2]; // [rsp+40h] [rbp-50h] BYREF
  _QWORD v9[8]; // [rsp+50h] [rbp-40h] BYREF

  qword_4FC08E0[0] = &unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4FC08E0[1]) &= 0xF000u;
  LODWORD(qword_4FC08E0[1]) = v0;
  qword_4FC08E0[11] = &qword_4FC08E0[15];
  qword_4FC08E0[12] = &qword_4FC08E0[15];
  qword_4FC08E0[9] = qword_4FA01C0;
  qword_4FC08E0[2] = 0;
  LOWORD(qword_4FC08E0[22]) = 256;
  qword_4FC08E0[21] = &unk_49E74E8;
  qword_4FC08E0[3] = 0;
  qword_4FC08E0[0] = &unk_49EEC70;
  qword_4FC08E0[23] = &unk_49EEDB0;
  qword_4FC08E0[4] = 0;
  qword_4FC08E0[5] = 0;
  qword_4FC08E0[6] = 0;
  qword_4FC08E0[7] = 0;
  qword_4FC08E0[8] = 0;
  qword_4FC08E0[10] = 0;
  qword_4FC08E0[13] = 4;
  LODWORD(qword_4FC08E0[14]) = 0;
  LOBYTE(qword_4FC08E0[19]) = 0;
  LOBYTE(qword_4FC08E0[20]) = 0;
  sub_16B8280(qword_4FC08E0, "cssa-coalesce", 13);
  LOWORD(qword_4FC08E0[22]) = 256;
  LOBYTE(qword_4FC08E0[20]) = 0;
  BYTE4(qword_4FC08E0[1]) = BYTE4(qword_4FC08E0[1]) & 0x9F | 0x20;
  sub_16B88A0(qword_4FC08E0);
  __cxa_atexit(sub_12EDEC0, qword_4FC08E0, &qword_4A427C0);
  qword_4FC0800[0] = &unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4FC0800[1]) &= 0xF000u;
  LODWORD(qword_4FC0800[1]) = v1;
  qword_4FC0800[11] = &qword_4FC0800[15];
  qword_4FC0800[12] = &qword_4FC0800[15];
  qword_4FC0800[2] = 0;
  qword_4FC0800[9] = qword_4FA01C0;
  qword_4FC0800[21] = &unk_49E74A8;
  qword_4FC0800[3] = 0;
  qword_4FC0800[4] = 0;
  qword_4FC0800[0] = &unk_49EEAF0;
  qword_4FC0800[5] = 0;
  qword_4FC0800[6] = 0;
  qword_4FC0800[23] = &unk_49EEE10;
  qword_4FC0800[7] = 0;
  qword_4FC0800[8] = 0;
  qword_4FC0800[10] = 0;
  qword_4FC0800[13] = 4;
  LODWORD(qword_4FC0800[14]) = 0;
  LOBYTE(qword_4FC0800[19]) = 0;
  LODWORD(qword_4FC0800[20]) = 0;
  BYTE4(qword_4FC0800[22]) = 1;
  LODWORD(qword_4FC0800[22]) = 0;
  sub_16B8280(qword_4FC0800, "cssa-verbosity", 14);
  LODWORD(qword_4FC0800[20]) = 0;
  BYTE4(qword_4FC0800[22]) = 1;
  LODWORD(qword_4FC0800[22]) = 0;
  BYTE4(qword_4FC0800[1]) = BYTE4(qword_4FC0800[1]) & 0x9F | 0x20;
  sub_16B88A0(qword_4FC0800);
  __cxa_atexit(sub_12EDE60, qword_4FC0800, &qword_4A427C0);
  qword_4FC0720 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC07D0 = 256;
  qword_4FC0730 = 0;
  word_4FC072C &= 0xF000u;
  qword_4FC07C8 = (__int64)&unk_49E74E8;
  qword_4FC0720 = (__int64)&unk_49EEC70;
  dword_4FC0728 = v2;
  qword_4FC07D8 = (__int64)&unk_49EEDB0;
  qword_4FC0768 = (__int64)qword_4FA01C0;
  qword_4FC0778 = (__int64)&unk_4FC0798;
  qword_4FC0780 = (__int64)&unk_4FC0798;
  qword_4FC0738 = 0;
  qword_4FC0740 = 0;
  qword_4FC0748 = 0;
  qword_4FC0750 = 0;
  qword_4FC0758 = 0;
  qword_4FC0760 = 0;
  qword_4FC0770 = 0;
  qword_4FC0788 = 4;
  dword_4FC0790 = 0;
  byte_4FC07B8 = 0;
  byte_4FC07C0 = 0;
  sub_16B8280((char *)&unk_4FC0798 - 120, "dump-before-cssa", 16);
  word_4FC07D0 = 256;
  byte_4FC07C0 = 0;
  LOBYTE(word_4FC072C) = word_4FC072C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC0720);
  __cxa_atexit(sub_12EDEC0, &qword_4FC0720, &qword_4A427C0);
  v3 = sub_16BAF20();
  v8[0] = v9;
  v4 = v3;
  sub_1CF0060(v8, "Controls which specific operands of phis in the module are coalesced");
  v6[0] = v7;
  sub_1CF0060(v6, "coalescing-counter");
  result = sub_14C9E50(v4, v6, v8);
  if ( (_QWORD *)v6[0] != v7 )
    result = j_j___libc_free_0(v6[0], v7[0] + 1LL);
  if ( (_QWORD *)v8[0] != v9 )
    return j_j___libc_free_0(v8[0], v9[0] + 1LL);
  return result;
}

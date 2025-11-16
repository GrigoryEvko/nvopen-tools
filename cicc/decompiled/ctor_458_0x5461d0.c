// Function: ctor_458
// Address: 0x5461d0
//
int ctor_458()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // rbx
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 v16; // [rsp+8h] [rbp-38h]

  qword_4FFE9A0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFE9F0 = 0x100000000LL;
  dword_4FFE9AC &= 0x8000u;
  word_4FFE9B0 = 0;
  qword_4FFE9B8 = 0;
  qword_4FFE9C0 = 0;
  dword_4FFE9A8 = v0;
  qword_4FFE9C8 = 0;
  qword_4FFE9D0 = 0;
  qword_4FFE9D8 = 0;
  qword_4FFE9E0 = 0;
  qword_4FFE9E8 = (__int64)&unk_4FFE9F8;
  qword_4FFEA00 = 0;
  qword_4FFEA08 = (__int64)&unk_4FFEA20;
  qword_4FFEA10 = 1;
  dword_4FFEA18 = 0;
  byte_4FFEA1C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FFE9F0;
  v3 = (unsigned int)qword_4FFE9F0 + 1LL;
  if ( v3 > HIDWORD(qword_4FFE9F0) )
  {
    sub_C8D5F0((char *)&unk_4FFE9F8 - 16, &unk_4FFE9F8, v3, 8);
    v2 = (unsigned int)qword_4FFE9F0;
  }
  *(_QWORD *)(qword_4FFE9E8 + 8 * v2) = v1;
  LODWORD(qword_4FFE9F0) = qword_4FFE9F0 + 1;
  qword_4FFEA28 = 0;
  qword_4FFEA30 = (__int64)&unk_49D9748;
  qword_4FFEA38 = 0;
  qword_4FFE9A0 = (__int64)&unk_49DC090;
  qword_4FFEA40 = (__int64)&unk_49DC1D0;
  qword_4FFEA60 = (__int64)nullsub_23;
  qword_4FFEA58 = (__int64)sub_984030;
  sub_C53080(&qword_4FFE9A0, "loop-prefetch-writes", 20);
  LOWORD(qword_4FFEA38) = 256;
  LOBYTE(qword_4FFEA28) = 0;
  qword_4FFE9D0 = 24;
  LOBYTE(dword_4FFE9AC) = dword_4FFE9AC & 0x9F | 0x20;
  qword_4FFE9C8 = (__int64)"Prefetch write addresses";
  sub_C53130(&qword_4FFE9A0);
  __cxa_atexit(sub_984900, &qword_4FFE9A0, &qword_4A427C0);
  qword_4FFE8C0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFE910 = 0x100000000LL;
  dword_4FFE8CC &= 0x8000u;
  word_4FFE8D0 = 0;
  qword_4FFE8D8 = 0;
  qword_4FFE8E0 = 0;
  dword_4FFE8C8 = v4;
  qword_4FFE8E8 = 0;
  qword_4FFE8F0 = 0;
  qword_4FFE8F8 = 0;
  qword_4FFE900 = 0;
  qword_4FFE908 = (__int64)&unk_4FFE918;
  qword_4FFE920 = 0;
  qword_4FFE928 = (__int64)&unk_4FFE940;
  qword_4FFE930 = 1;
  dword_4FFE938 = 0;
  byte_4FFE93C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FFE910;
  v7 = (unsigned int)qword_4FFE910 + 1LL;
  if ( v7 > HIDWORD(qword_4FFE910) )
  {
    sub_C8D5F0((char *)&unk_4FFE918 - 16, &unk_4FFE918, v7, 8);
    v6 = (unsigned int)qword_4FFE910;
  }
  *(_QWORD *)(qword_4FFE908 + 8 * v6) = v5;
  qword_4FFE950 = (__int64)&unk_49D9728;
  qword_4FFE8C0 = (__int64)&unk_49DBF10;
  qword_4FFE960 = (__int64)&unk_49DC290;
  LODWORD(qword_4FFE910) = qword_4FFE910 + 1;
  qword_4FFE980 = (__int64)nullsub_24;
  qword_4FFE948 = 0;
  qword_4FFE978 = (__int64)sub_984050;
  qword_4FFE958 = 0;
  sub_C53080(&qword_4FFE8C0, "prefetch-distance", 17);
  qword_4FFE8F0 = 40;
  qword_4FFE8E8 = (__int64)"Number of instructions to prefetch ahead";
  LOBYTE(dword_4FFE8CC) = dword_4FFE8CC & 0x9F | 0x20;
  sub_C53130(&qword_4FFE8C0);
  __cxa_atexit(sub_984970, &qword_4FFE8C0, &qword_4A427C0);
  qword_4FFE7E0 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FFE85C = 1;
  qword_4FFE830 = 0x100000000LL;
  dword_4FFE7EC &= 0x8000u;
  qword_4FFE828 = (__int64)&unk_4FFE838;
  qword_4FFE7F8 = 0;
  qword_4FFE800 = 0;
  dword_4FFE7E8 = v8;
  word_4FFE7F0 = 0;
  qword_4FFE808 = 0;
  qword_4FFE810 = 0;
  qword_4FFE818 = 0;
  qword_4FFE820 = 0;
  qword_4FFE840 = 0;
  qword_4FFE848 = (__int64)&unk_4FFE860;
  qword_4FFE850 = 1;
  dword_4FFE858 = 0;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4FFE830;
  if ( (unsigned __int64)(unsigned int)qword_4FFE830 + 1 > HIDWORD(qword_4FFE830) )
  {
    v16 = v9;
    sub_C8D5F0((char *)&unk_4FFE838 - 16, &unk_4FFE838, (unsigned int)qword_4FFE830 + 1LL, 8);
    v10 = (unsigned int)qword_4FFE830;
    v9 = v16;
  }
  *(_QWORD *)(qword_4FFE828 + 8 * v10) = v9;
  qword_4FFE870 = (__int64)&unk_49D9728;
  qword_4FFE7E0 = (__int64)&unk_49DBF10;
  qword_4FFE880 = (__int64)&unk_49DC290;
  LODWORD(qword_4FFE830) = qword_4FFE830 + 1;
  qword_4FFE8A0 = (__int64)nullsub_24;
  qword_4FFE868 = 0;
  qword_4FFE898 = (__int64)sub_984050;
  qword_4FFE878 = 0;
  sub_C53080(&qword_4FFE7E0, "min-prefetch-stride", 19);
  qword_4FFE810 = 28;
  qword_4FFE808 = (__int64)"Min stride to add prefetches";
  LOBYTE(dword_4FFE7EC) = dword_4FFE7EC & 0x9F | 0x20;
  sub_C53130(&qword_4FFE7E0);
  __cxa_atexit(sub_984970, &qword_4FFE7E0, &qword_4A427C0);
  qword_4FFE700 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FFE70C &= 0x8000u;
  word_4FFE710 = 0;
  qword_4FFE750 = 0x100000000LL;
  qword_4FFE748 = (__int64)&unk_4FFE758;
  qword_4FFE718 = 0;
  qword_4FFE720 = 0;
  dword_4FFE708 = v11;
  qword_4FFE728 = 0;
  qword_4FFE730 = 0;
  qword_4FFE738 = 0;
  qword_4FFE740 = 0;
  qword_4FFE760 = 0;
  qword_4FFE768 = (__int64)&unk_4FFE780;
  qword_4FFE770 = 1;
  dword_4FFE778 = 0;
  byte_4FFE77C = 1;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_4FFE750;
  v14 = (unsigned int)qword_4FFE750 + 1LL;
  if ( v14 > HIDWORD(qword_4FFE750) )
  {
    sub_C8D5F0((char *)&unk_4FFE758 - 16, &unk_4FFE758, v14, 8);
    v13 = (unsigned int)qword_4FFE750;
  }
  *(_QWORD *)(qword_4FFE748 + 8 * v13) = v12;
  qword_4FFE790 = (__int64)&unk_49D9728;
  qword_4FFE700 = (__int64)&unk_49DBF10;
  qword_4FFE7A0 = (__int64)&unk_49DC290;
  LODWORD(qword_4FFE750) = qword_4FFE750 + 1;
  qword_4FFE7C0 = (__int64)nullsub_24;
  qword_4FFE788 = 0;
  qword_4FFE7B8 = (__int64)sub_984050;
  qword_4FFE798 = 0;
  sub_C53080(&qword_4FFE700, "max-prefetch-iters-ahead", 24);
  qword_4FFE730 = 42;
  qword_4FFE728 = (__int64)"Max number of iterations to prefetch ahead";
  LOBYTE(dword_4FFE70C) = dword_4FFE70C & 0x9F | 0x20;
  sub_C53130(&qword_4FFE700);
  return __cxa_atexit(sub_984970, &qword_4FFE700, &qword_4A427C0);
}

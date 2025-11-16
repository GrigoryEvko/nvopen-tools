// Function: ctor_407
// Address: 0x52c620
//
int ctor_407()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // edx
  __int64 v18; // rax
  __int64 v19; // rdx
  int v20; // edx
  __int64 v21; // rbx
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v25; // [rsp+8h] [rbp-38h]
  __int64 v26; // [rsp+8h] [rbp-38h]
  __int64 v27; // [rsp+8h] [rbp-38h]
  __int64 v28; // [rsp+8h] [rbp-38h]

  qword_4FECA40 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FECA90 = 0x100000000LL;
  word_4FECA50 = 0;
  dword_4FECA4C &= 0x8000u;
  qword_4FECA58 = 0;
  qword_4FECA60 = 0;
  dword_4FECA48 = v0;
  qword_4FECA68 = 0;
  qword_4FECA70 = 0;
  qword_4FECA78 = 0;
  qword_4FECA80 = 0;
  qword_4FECA88 = (__int64)&unk_4FECA98;
  qword_4FECAA0 = 0;
  qword_4FECAA8 = (__int64)&unk_4FECAC0;
  qword_4FECAB0 = 1;
  dword_4FECAB8 = 0;
  byte_4FECABC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FECA90;
  v3 = (unsigned int)qword_4FECA90 + 1LL;
  if ( v3 > HIDWORD(qword_4FECA90) )
  {
    sub_C8D5F0((char *)&unk_4FECA98 - 16, &unk_4FECA98, v3, 8);
    v2 = (unsigned int)qword_4FECA90;
  }
  *(_QWORD *)(qword_4FECA88 + 8 * v2) = v1;
  qword_4FECAD0 = (__int64)&unk_49D9728;
  LODWORD(qword_4FECA90) = qword_4FECA90 + 1;
  qword_4FECAC8 = 0;
  qword_4FECA40 = (__int64)&unk_49DBF10;
  qword_4FECAD8 = 0;
  qword_4FECAE0 = (__int64)&unk_49DC290;
  qword_4FECB00 = (__int64)nullsub_24;
  qword_4FECAF8 = (__int64)sub_984050;
  sub_C53080(&qword_4FECA40, "pgo-memop-count-threshold", 25);
  LODWORD(qword_4FECAC8) = 1000;
  BYTE4(qword_4FECAD8) = 1;
  LODWORD(qword_4FECAD8) = 1000;
  qword_4FECA70 = 52;
  LOBYTE(dword_4FECA4C) = dword_4FECA4C & 0x9F | 0x20;
  qword_4FECA68 = (__int64)"The minimum count to optimize memory intrinsic calls";
  sub_C53130(&qword_4FECA40);
  __cxa_atexit(sub_984970, &qword_4FECA40, &qword_4A427C0);
  qword_4FEC960 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FEC9B0 = 0x100000000LL;
  dword_4FEC96C &= 0x8000u;
  word_4FEC970 = 0;
  qword_4FEC978 = 0;
  qword_4FEC980 = 0;
  dword_4FEC968 = v4;
  qword_4FEC988 = 0;
  qword_4FEC990 = 0;
  qword_4FEC998 = 0;
  qword_4FEC9A0 = 0;
  qword_4FEC9A8 = (__int64)&unk_4FEC9B8;
  qword_4FEC9C0 = 0;
  qword_4FEC9C8 = (__int64)&unk_4FEC9E0;
  qword_4FEC9D0 = 1;
  dword_4FEC9D8 = 0;
  byte_4FEC9DC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FEC9B0;
  v7 = (unsigned int)qword_4FEC9B0 + 1LL;
  if ( v7 > HIDWORD(qword_4FEC9B0) )
  {
    sub_C8D5F0((char *)&unk_4FEC9B8 - 16, &unk_4FEC9B8, v7, 8);
    v6 = (unsigned int)qword_4FEC9B0;
  }
  *(_QWORD *)(qword_4FEC9A8 + 8 * v6) = v5;
  LODWORD(qword_4FEC9B0) = qword_4FEC9B0 + 1;
  qword_4FEC9E8 = 0;
  qword_4FEC9F0 = (__int64)&unk_49D9748;
  qword_4FEC9F8 = 0;
  qword_4FEC960 = (__int64)&unk_49DC090;
  qword_4FECA00 = (__int64)&unk_49DC1D0;
  qword_4FECA20 = (__int64)nullsub_23;
  qword_4FECA18 = (__int64)sub_984030;
  sub_C53080(&qword_4FEC960, "disable-memop-opt", 17);
  LOWORD(qword_4FEC9F8) = 256;
  LOBYTE(qword_4FEC9E8) = 0;
  qword_4FEC990 = 16;
  LOBYTE(dword_4FEC96C) = dword_4FEC96C & 0x9F | 0x20;
  qword_4FEC988 = (__int64)"Disable optimize";
  sub_C53130(&qword_4FEC960);
  __cxa_atexit(sub_984900, &qword_4FEC960, &qword_4A427C0);
  qword_4FEC880 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FEC8D0 = 0x100000000LL;
  dword_4FEC88C &= 0x8000u;
  word_4FEC890 = 0;
  qword_4FEC898 = 0;
  qword_4FEC8A0 = 0;
  dword_4FEC888 = v8;
  qword_4FEC8A8 = 0;
  qword_4FEC8B0 = 0;
  qword_4FEC8B8 = 0;
  qword_4FEC8C0 = 0;
  qword_4FEC8C8 = (__int64)&unk_4FEC8D8;
  qword_4FEC8E0 = 0;
  qword_4FEC8E8 = (__int64)&unk_4FEC900;
  qword_4FEC8F0 = 1;
  dword_4FEC8F8 = 0;
  byte_4FEC8FC = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4FEC8D0;
  if ( (unsigned __int64)(unsigned int)qword_4FEC8D0 + 1 > HIDWORD(qword_4FEC8D0) )
  {
    v25 = v9;
    sub_C8D5F0((char *)&unk_4FEC8D8 - 16, &unk_4FEC8D8, (unsigned int)qword_4FEC8D0 + 1LL, 8);
    v10 = (unsigned int)qword_4FEC8D0;
    v9 = v25;
  }
  *(_QWORD *)(qword_4FEC8C8 + 8 * v10) = v9;
  qword_4FEC910 = (__int64)&unk_49D9728;
  LODWORD(qword_4FEC8D0) = qword_4FEC8D0 + 1;
  qword_4FEC908 = 0;
  qword_4FEC880 = (__int64)&unk_49DBF10;
  qword_4FEC918 = 0;
  qword_4FEC920 = (__int64)&unk_49DC290;
  qword_4FEC940 = (__int64)nullsub_24;
  qword_4FEC938 = (__int64)sub_984050;
  sub_C53080(&qword_4FEC880, "pgo-memop-percent-threshold", 27);
  LODWORD(qword_4FEC908) = 40;
  BYTE4(qword_4FEC918) = 1;
  LODWORD(qword_4FEC918) = 40;
  qword_4FEC8B0 = 68;
  LOBYTE(dword_4FEC88C) = dword_4FEC88C & 0x9F | 0x20;
  qword_4FEC8A8 = (__int64)"The percentage threshold for the memory intrinsic calls optimization";
  sub_C53130(&qword_4FEC880);
  __cxa_atexit(sub_984970, &qword_4FEC880, &qword_4A427C0);
  qword_4FEC7A0 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FEC7F0 = 0x100000000LL;
  word_4FEC7B0 = 0;
  dword_4FEC7AC &= 0x8000u;
  qword_4FEC7B8 = 0;
  qword_4FEC7C0 = 0;
  dword_4FEC7A8 = v11;
  qword_4FEC7C8 = 0;
  qword_4FEC7D0 = 0;
  qword_4FEC7D8 = 0;
  qword_4FEC7E0 = 0;
  qword_4FEC7E8 = (__int64)&unk_4FEC7F8;
  qword_4FEC800 = 0;
  qword_4FEC808 = (__int64)&unk_4FEC820;
  qword_4FEC810 = 1;
  dword_4FEC818 = 0;
  byte_4FEC81C = 1;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_4FEC7F0;
  if ( (unsigned __int64)(unsigned int)qword_4FEC7F0 + 1 > HIDWORD(qword_4FEC7F0) )
  {
    v26 = v12;
    sub_C8D5F0((char *)&unk_4FEC7F8 - 16, &unk_4FEC7F8, (unsigned int)qword_4FEC7F0 + 1LL, 8);
    v13 = (unsigned int)qword_4FEC7F0;
    v12 = v26;
  }
  *(_QWORD *)(qword_4FEC7E8 + 8 * v13) = v12;
  qword_4FEC830 = (__int64)&unk_49D9728;
  LODWORD(qword_4FEC7F0) = qword_4FEC7F0 + 1;
  qword_4FEC828 = 0;
  qword_4FEC7A0 = (__int64)&unk_49DBF10;
  qword_4FEC838 = 0;
  qword_4FEC840 = (__int64)&unk_49DC290;
  qword_4FEC860 = (__int64)nullsub_24;
  qword_4FEC858 = (__int64)sub_984050;
  sub_C53080(&qword_4FEC7A0, "pgo-memop-max-version", 21);
  LODWORD(qword_4FEC828) = 3;
  BYTE4(qword_4FEC838) = 1;
  LODWORD(qword_4FEC838) = 3;
  qword_4FEC7D0 = 57;
  LOBYTE(dword_4FEC7AC) = dword_4FEC7AC & 0x9F | 0x20;
  qword_4FEC7C8 = (__int64)"The max version for the optimized memory  intrinsic calls";
  sub_C53130(&qword_4FEC7A0);
  __cxa_atexit(sub_984970, &qword_4FEC7A0, &qword_4A427C0);
  qword_4FEC6C0 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FEC710 = 0x100000000LL;
  dword_4FEC6CC &= 0x8000u;
  word_4FEC6D0 = 0;
  qword_4FEC6D8 = 0;
  qword_4FEC6E0 = 0;
  dword_4FEC6C8 = v14;
  qword_4FEC6E8 = 0;
  qword_4FEC6F0 = 0;
  qword_4FEC6F8 = 0;
  qword_4FEC700 = 0;
  qword_4FEC708 = (__int64)&unk_4FEC718;
  qword_4FEC720 = 0;
  qword_4FEC728 = (__int64)&unk_4FEC740;
  qword_4FEC730 = 1;
  dword_4FEC738 = 0;
  byte_4FEC73C = 1;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_4FEC710;
  if ( (unsigned __int64)(unsigned int)qword_4FEC710 + 1 > HIDWORD(qword_4FEC710) )
  {
    v27 = v15;
    sub_C8D5F0((char *)&unk_4FEC718 - 16, &unk_4FEC718, (unsigned int)qword_4FEC710 + 1LL, 8);
    v16 = (unsigned int)qword_4FEC710;
    v15 = v27;
  }
  *(_QWORD *)(qword_4FEC708 + 8 * v16) = v15;
  LODWORD(qword_4FEC710) = qword_4FEC710 + 1;
  qword_4FEC748 = 0;
  qword_4FEC750 = (__int64)&unk_49D9748;
  qword_4FEC758 = 0;
  qword_4FEC6C0 = (__int64)&unk_49DC090;
  qword_4FEC760 = (__int64)&unk_49DC1D0;
  qword_4FEC780 = (__int64)nullsub_23;
  qword_4FEC778 = (__int64)sub_984030;
  sub_C53080(&qword_4FEC6C0, "pgo-memop-scale-count", 21);
  LOWORD(qword_4FEC758) = 257;
  LOBYTE(qword_4FEC748) = 1;
  qword_4FEC6F0 = 62;
  LOBYTE(dword_4FEC6CC) = dword_4FEC6CC & 0x9F | 0x20;
  qword_4FEC6E8 = (__int64)"Scale the memop size counts using the basic  block count value";
  sub_C53130(&qword_4FEC6C0);
  __cxa_atexit(sub_984900, &qword_4FEC6C0, &qword_4A427C0);
  qword_4FEC5E0 = &unk_49DC150;
  v17 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4FEC5EC = word_4FEC5EC & 0x8000;
  unk_4FEC5F0 = 0;
  qword_4FEC628[1] = 0x100000000LL;
  unk_4FEC5E8 = v17;
  qword_4FEC628[0] = &qword_4FEC628[2];
  unk_4FEC5F8 = 0;
  unk_4FEC600 = 0;
  unk_4FEC608 = 0;
  unk_4FEC610 = 0;
  unk_4FEC618 = 0;
  unk_4FEC620 = 0;
  qword_4FEC628[3] = 0;
  qword_4FEC628[4] = &qword_4FEC628[7];
  qword_4FEC628[5] = 1;
  LODWORD(qword_4FEC628[6]) = 0;
  BYTE4(qword_4FEC628[6]) = 1;
  v18 = sub_C57470();
  v19 = LODWORD(qword_4FEC628[1]);
  if ( (unsigned __int64)LODWORD(qword_4FEC628[1]) + 1 > HIDWORD(qword_4FEC628[1]) )
  {
    v28 = v18;
    sub_C8D5F0(qword_4FEC628, &qword_4FEC628[2], LODWORD(qword_4FEC628[1]) + 1LL, 8);
    v19 = LODWORD(qword_4FEC628[1]);
    v18 = v28;
  }
  *(_QWORD *)(qword_4FEC628[0] + 8 * v19) = v18;
  ++LODWORD(qword_4FEC628[1]);
  qword_4FEC628[8] = 0;
  qword_4FEC628[9] = &unk_49D9748;
  qword_4FEC628[10] = 0;
  qword_4FEC5E0 = &unk_49DC090;
  qword_4FEC628[11] = &unk_49DC1D0;
  qword_4FEC628[15] = nullsub_23;
  qword_4FEC628[14] = sub_984030;
  sub_C53080(&qword_4FEC5E0, "pgo-memop-optimize-memcmp-bcmp", 30);
  LOBYTE(qword_4FEC628[8]) = 1;
  LOWORD(qword_4FEC628[10]) = 257;
  unk_4FEC610 = 37;
  LOBYTE(word_4FEC5EC) = word_4FEC5EC & 0x9F | 0x20;
  unk_4FEC608 = "Size-specialize memcmp and bcmp calls";
  sub_C53130(&qword_4FEC5E0);
  __cxa_atexit(sub_984900, &qword_4FEC5E0, &qword_4A427C0);
  qword_4FEC500 = (__int64)&unk_49DC150;
  v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FEC57C = 1;
  qword_4FEC550 = 0x100000000LL;
  dword_4FEC50C &= 0x8000u;
  qword_4FEC518 = 0;
  qword_4FEC520 = 0;
  qword_4FEC528 = 0;
  dword_4FEC508 = v20;
  word_4FEC510 = 0;
  qword_4FEC530 = 0;
  qword_4FEC538 = 0;
  qword_4FEC540 = 0;
  qword_4FEC548 = (__int64)&unk_4FEC558;
  qword_4FEC560 = 0;
  qword_4FEC568 = (__int64)&unk_4FEC580;
  qword_4FEC570 = 1;
  dword_4FEC578 = 0;
  v21 = sub_C57470();
  v22 = (unsigned int)qword_4FEC550;
  v23 = (unsigned int)qword_4FEC550 + 1LL;
  if ( v23 > HIDWORD(qword_4FEC550) )
  {
    sub_C8D5F0((char *)&unk_4FEC558 - 16, &unk_4FEC558, v23, 8);
    v22 = (unsigned int)qword_4FEC550;
  }
  *(_QWORD *)(qword_4FEC548 + 8 * v22) = v21;
  qword_4FEC590 = (__int64)&unk_49D9728;
  LODWORD(qword_4FEC550) = qword_4FEC550 + 1;
  qword_4FEC588 = 0;
  qword_4FEC500 = (__int64)&unk_49DBF10;
  qword_4FEC598 = 0;
  qword_4FEC5A0 = (__int64)&unk_49DC290;
  qword_4FEC5C0 = (__int64)nullsub_24;
  qword_4FEC5B8 = (__int64)sub_984050;
  sub_C53080(&qword_4FEC500, "memop-value-prof-max-opt-size", 29);
  LODWORD(qword_4FEC588) = 128;
  BYTE4(qword_4FEC598) = 1;
  LODWORD(qword_4FEC598) = 128;
  qword_4FEC530 = 37;
  LOBYTE(dword_4FEC50C) = dword_4FEC50C & 0x9F | 0x20;
  qword_4FEC528 = (__int64)"Optimize the memop size <= this value";
  sub_C53130(&qword_4FEC500);
  return __cxa_atexit(sub_984970, &qword_4FEC500, &qword_4A427C0);
}

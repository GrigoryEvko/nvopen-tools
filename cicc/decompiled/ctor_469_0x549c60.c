// Function: ctor_469
// Address: 0x549c60
//
int ctor_469()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5000800 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_500087C = 1;
  qword_5000850 = 0x100000000LL;
  dword_500080C &= 0x8000u;
  qword_5000818 = 0;
  qword_5000820 = 0;
  qword_5000828 = 0;
  dword_5000808 = v0;
  word_5000810 = 0;
  qword_5000830 = 0;
  qword_5000838 = 0;
  qword_5000840 = 0;
  qword_5000848 = (__int64)&unk_5000858;
  qword_5000860 = 0;
  qword_5000868 = (__int64)&unk_5000880;
  qword_5000870 = 1;
  dword_5000878 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5000850;
  v3 = (unsigned int)qword_5000850 + 1LL;
  if ( v3 > HIDWORD(qword_5000850) )
  {
    sub_C8D5F0((char *)&unk_5000858 - 16, &unk_5000858, v3, 8);
    v2 = (unsigned int)qword_5000850;
  }
  *(_QWORD *)(qword_5000848 + 8 * v2) = v1;
  qword_5000890 = (__int64)&unk_49D9728;
  LODWORD(qword_5000850) = qword_5000850 + 1;
  qword_5000888 = 0;
  qword_5000800 = (__int64)&unk_49DBF10;
  qword_50008A0 = (__int64)&unk_49DC290;
  qword_5000898 = 0;
  qword_50008C0 = (__int64)nullsub_24;
  qword_50008B8 = (__int64)sub_984050;
  sub_C53080(&qword_5000800, "sink-freq-percent-threshold", 27);
  LODWORD(qword_5000888) = 90;
  BYTE4(qword_5000898) = 1;
  LODWORD(qword_5000898) = 90;
  qword_5000830 = 101;
  LOBYTE(dword_500080C) = dword_500080C & 0x9F | 0x20;
  qword_5000828 = (__int64)"Do not sink instructions that require cloning unless they execute less than this percent of the time.";
  sub_C53130(&qword_5000800);
  __cxa_atexit(sub_984970, &qword_5000800, &qword_4A427C0);
  qword_5000720 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_500072C &= 0x8000u;
  word_5000730 = 0;
  qword_5000770 = 0x100000000LL;
  qword_5000738 = 0;
  qword_5000740 = 0;
  qword_5000748 = 0;
  dword_5000728 = v4;
  qword_5000750 = 0;
  qword_5000758 = 0;
  qword_5000760 = 0;
  qword_5000768 = (__int64)&unk_5000778;
  qword_5000780 = 0;
  qword_5000788 = (__int64)&unk_50007A0;
  qword_5000790 = 1;
  dword_5000798 = 0;
  byte_500079C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5000770;
  v7 = (unsigned int)qword_5000770 + 1LL;
  if ( v7 > HIDWORD(qword_5000770) )
  {
    sub_C8D5F0((char *)&unk_5000778 - 16, &unk_5000778, v7, 8);
    v6 = (unsigned int)qword_5000770;
  }
  *(_QWORD *)(qword_5000768 + 8 * v6) = v5;
  qword_50007B0 = (__int64)&unk_49D9728;
  LODWORD(qword_5000770) = qword_5000770 + 1;
  qword_50007A8 = 0;
  qword_5000720 = (__int64)&unk_49DBF10;
  qword_50007C0 = (__int64)&unk_49DC290;
  qword_50007B8 = 0;
  qword_50007E0 = (__int64)nullsub_24;
  qword_50007D8 = (__int64)sub_984050;
  sub_C53080(&qword_5000720, "max-uses-for-sinking", 20);
  LODWORD(qword_50007A8) = 30;
  BYTE4(qword_50007B8) = 1;
  LODWORD(qword_50007B8) = 30;
  qword_5000750 = 49;
  LOBYTE(dword_500072C) = dword_500072C & 0x9F | 0x20;
  qword_5000748 = (__int64)"Do not sink instructions that have too many uses.";
  sub_C53130(&qword_5000720);
  return __cxa_atexit(sub_984970, &qword_5000720, &qword_4A427C0);
}

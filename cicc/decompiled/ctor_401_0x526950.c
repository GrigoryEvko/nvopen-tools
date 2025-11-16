// Function: ctor_401
// Address: 0x526950
//
int ctor_401()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_4FE7800 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FE787C = 1;
  qword_4FE7850 = 0x100000000LL;
  dword_4FE780C &= 0x8000u;
  qword_4FE7818 = 0;
  qword_4FE7820 = 0;
  qword_4FE7828 = 0;
  dword_4FE7808 = v0;
  word_4FE7810 = 0;
  qword_4FE7830 = 0;
  qword_4FE7838 = 0;
  qword_4FE7840 = 0;
  qword_4FE7848 = (__int64)&unk_4FE7858;
  qword_4FE7860 = 0;
  qword_4FE7868 = (__int64)&unk_4FE7880;
  qword_4FE7870 = 1;
  dword_4FE7878 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FE7850;
  v3 = (unsigned int)qword_4FE7850 + 1LL;
  if ( v3 > HIDWORD(qword_4FE7850) )
  {
    sub_C8D5F0((char *)&unk_4FE7858 - 16, &unk_4FE7858, v3, 8);
    v2 = (unsigned int)qword_4FE7850;
  }
  *(_QWORD *)(qword_4FE7848 + 8 * v2) = v1;
  LODWORD(qword_4FE7850) = qword_4FE7850 + 1;
  qword_4FE7888 = 0;
  qword_4FE7890 = (__int64)&unk_49DA090;
  qword_4FE7898 = 0;
  qword_4FE7800 = (__int64)&unk_49DBF90;
  qword_4FE78A0 = (__int64)&unk_49DC230;
  qword_4FE78C0 = (__int64)nullsub_58;
  qword_4FE78B8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FE7800, "lower-allow-check-percentile-cutoff-hot", 39);
  qword_4FE7830 = 22;
  qword_4FE7828 = (__int64)"Hot percentile cutoff.";
  sub_C53130(&qword_4FE7800);
  __cxa_atexit(sub_B2B680, &qword_4FE7800, &qword_4A427C0);
  qword_4FE7720 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FE772C &= 0x8000u;
  word_4FE7730 = 0;
  qword_4FE7770 = 0x100000000LL;
  qword_4FE7738 = 0;
  qword_4FE7740 = 0;
  qword_4FE7748 = 0;
  dword_4FE7728 = v4;
  qword_4FE7750 = 0;
  qword_4FE7758 = 0;
  qword_4FE7760 = 0;
  qword_4FE7768 = (__int64)&unk_4FE7778;
  qword_4FE7780 = 0;
  qword_4FE7788 = (__int64)&unk_4FE77A0;
  qword_4FE7790 = 1;
  dword_4FE7798 = 0;
  byte_4FE779C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FE7770;
  v7 = (unsigned int)qword_4FE7770 + 1LL;
  if ( v7 > HIDWORD(qword_4FE7770) )
  {
    sub_C8D5F0((char *)&unk_4FE7778 - 16, &unk_4FE7778, v7, 8);
    v6 = (unsigned int)qword_4FE7770;
  }
  *(_QWORD *)(qword_4FE7768 + 8 * v6) = v5;
  LODWORD(qword_4FE7770) = qword_4FE7770 + 1;
  qword_4FE77A8 = 0;
  qword_4FE77B0 = (__int64)&unk_49E5940;
  qword_4FE77B8 = 0;
  qword_4FE7720 = (__int64)&unk_49E5960;
  qword_4FE77C0 = (__int64)&unk_49DC320;
  qword_4FE77E0 = (__int64)nullsub_385;
  qword_4FE77D8 = (__int64)sub_1038930;
  sub_C53080(&qword_4FE7720, "lower-allow-check-random-rate", 29);
  qword_4FE7750 = 80;
  qword_4FE7748 = (__int64)"Probability value in the range [0.0, 1.0] of unconditional pseudo-random checks.";
  sub_C53130(&qword_4FE7720);
  return __cxa_atexit(sub_1038DB0, &qword_4FE7720, &qword_4A427C0);
}

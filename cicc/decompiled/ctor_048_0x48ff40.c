// Function: ctor_048
// Address: 0x48ff40
//
int ctor_048()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // r15
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-38h]

  qword_4F868E0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F86930 = 0x100000000LL;
  dword_4F868EC &= 0x8000u;
  word_4F868F0 = 0;
  qword_4F868F8 = 0;
  qword_4F86900 = 0;
  dword_4F868E8 = v0;
  qword_4F86908 = 0;
  qword_4F86910 = 0;
  qword_4F86918 = 0;
  qword_4F86920 = 0;
  qword_4F86928 = (__int64)&unk_4F86938;
  qword_4F86940 = 0;
  qword_4F86948 = (__int64)&unk_4F86960;
  qword_4F86950 = 1;
  dword_4F86958 = 0;
  byte_4F8695C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F86930;
  v3 = (unsigned int)qword_4F86930 + 1LL;
  if ( v3 > HIDWORD(qword_4F86930) )
  {
    sub_C8D5F0((char *)&unk_4F86938 - 16, &unk_4F86938, v3, 8);
    v2 = (unsigned int)qword_4F86930;
  }
  *(_QWORD *)(qword_4F86928 + 8 * v2) = v1;
  qword_4F86970 = (__int64)&unk_49D9748;
  qword_4F868E0 = (__int64)&unk_49DC090;
  qword_4F86980 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F86930) = qword_4F86930 + 1;
  qword_4F869A0 = (__int64)nullsub_23;
  qword_4F86968 = 0;
  qword_4F86998 = (__int64)sub_984030;
  qword_4F86978 = 0;
  sub_C53080(&qword_4F868E0, "basic-aa-recphi", 15);
  LOWORD(qword_4F86978) = 257;
  LOBYTE(qword_4F86968) = 1;
  LOBYTE(dword_4F868EC) = dword_4F868EC & 0x9F | 0x20;
  sub_C53130(&qword_4F868E0);
  __cxa_atexit(sub_984900, &qword_4F868E0, &qword_4A427C0);
  qword_4F86800 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F86850 = 0x100000000LL;
  dword_4F8680C &= 0x8000u;
  qword_4F86848 = (__int64)&unk_4F86858;
  word_4F86810 = 0;
  qword_4F86818 = 0;
  dword_4F86808 = v4;
  qword_4F86820 = 0;
  qword_4F86828 = 0;
  qword_4F86830 = 0;
  qword_4F86838 = 0;
  qword_4F86840 = 0;
  qword_4F86860 = 0;
  qword_4F86868 = (__int64)&unk_4F86880;
  qword_4F86870 = 1;
  dword_4F86878 = 0;
  byte_4F8687C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F86850;
  if ( (unsigned __int64)(unsigned int)qword_4F86850 + 1 > HIDWORD(qword_4F86850) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_4F86858 - 16, &unk_4F86858, (unsigned int)qword_4F86850 + 1LL, 8);
    v6 = (unsigned int)qword_4F86850;
    v5 = v12;
  }
  *(_QWORD *)(qword_4F86848 + 8 * v6) = v5;
  qword_4F86890 = (__int64)&unk_49D9748;
  qword_4F86800 = (__int64)&unk_49DC090;
  qword_4F868A0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F86850) = qword_4F86850 + 1;
  qword_4F868C0 = (__int64)nullsub_23;
  qword_4F86888 = 0;
  qword_4F868B8 = (__int64)sub_984030;
  qword_4F86898 = 0;
  sub_C53080(&qword_4F86800, "basic-aa-full-recphi", 20);
  LOWORD(qword_4F86898) = 257;
  LOBYTE(qword_4F86888) = 1;
  LOBYTE(dword_4F8680C) = dword_4F8680C & 0x9F | 0x20;
  sub_C53130(&qword_4F86800);
  __cxa_atexit(sub_984900, &qword_4F86800, &qword_4A427C0);
  qword_4F86720 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F86770 = 0x100000000LL;
  dword_4F8672C &= 0x8000u;
  word_4F86730 = 0;
  qword_4F86768 = (__int64)&unk_4F86778;
  qword_4F86738 = 0;
  dword_4F86728 = v7;
  qword_4F86740 = 0;
  qword_4F86748 = 0;
  qword_4F86750 = 0;
  qword_4F86758 = 0;
  qword_4F86760 = 0;
  qword_4F86780 = 0;
  qword_4F86788 = (__int64)&unk_4F867A0;
  qword_4F86790 = 1;
  dword_4F86798 = 0;
  byte_4F8679C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4F86770;
  v10 = (unsigned int)qword_4F86770 + 1LL;
  if ( v10 > HIDWORD(qword_4F86770) )
  {
    sub_C8D5F0((char *)&unk_4F86778 - 16, &unk_4F86778, v10, 8);
    v9 = (unsigned int)qword_4F86770;
  }
  *(_QWORD *)(qword_4F86768 + 8 * v9) = v8;
  qword_4F867B0 = (__int64)&unk_49D9748;
  qword_4F86720 = (__int64)&unk_49DC090;
  qword_4F867C0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F86770) = qword_4F86770 + 1;
  qword_4F867E0 = (__int64)nullsub_23;
  qword_4F867A8 = 0;
  qword_4F867D8 = (__int64)sub_984030;
  qword_4F867B8 = 0;
  sub_C53080(&qword_4F86720, "basic-aa-separate-storage", 25);
  LOBYTE(qword_4F867A8) = 1;
  LOBYTE(dword_4F8672C) = dword_4F8672C & 0x9F | 0x20;
  LOWORD(qword_4F867B8) = 257;
  sub_C53130(&qword_4F86720);
  return __cxa_atexit(sub_984900, &qword_4F86720, &qword_4A427C0);
}

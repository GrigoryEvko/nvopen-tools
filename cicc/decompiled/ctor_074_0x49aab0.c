// Function: ctor_074
// Address: 0x49aab0
//
int ctor_074()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // rbx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v19; // [rsp+8h] [rbp-38h]
  __int64 v20; // [rsp+8h] [rbp-38h]

  qword_4F8D800 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8D850 = 0x100000000LL;
  dword_4F8D80C &= 0x8000u;
  word_4F8D810 = 0;
  qword_4F8D818 = 0;
  qword_4F8D820 = 0;
  dword_4F8D808 = v0;
  qword_4F8D828 = 0;
  qword_4F8D830 = 0;
  qword_4F8D838 = 0;
  qword_4F8D840 = 0;
  qword_4F8D848 = (__int64)&unk_4F8D858;
  qword_4F8D860 = 0;
  qword_4F8D868 = (__int64)&unk_4F8D880;
  qword_4F8D870 = 1;
  dword_4F8D878 = 0;
  byte_4F8D87C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F8D850;
  v3 = (unsigned int)qword_4F8D850 + 1LL;
  if ( v3 > HIDWORD(qword_4F8D850) )
  {
    sub_C8D5F0((char *)&unk_4F8D858 - 16, &unk_4F8D858, v3, 8);
    v2 = (unsigned int)qword_4F8D850;
  }
  *(_QWORD *)(qword_4F8D848 + 8 * v2) = v1;
  qword_4F8D890 = (__int64)&unk_49DA090;
  qword_4F8D800 = (__int64)&unk_49DBF90;
  qword_4F8D8A0 = (__int64)&unk_49DC230;
  LODWORD(qword_4F8D850) = qword_4F8D850 + 1;
  qword_4F8D8C0 = (__int64)nullsub_58;
  qword_4F8D888 = 0;
  qword_4F8D8B8 = (__int64)sub_B2B5F0;
  qword_4F8D898 = 0;
  sub_C53080(&qword_4F8D800, "fca-size", 8);
  LODWORD(qword_4F8D888) = 8;
  BYTE4(qword_4F8D898) = 1;
  LODWORD(qword_4F8D898) = 8;
  qword_4F8D830 = 47;
  LOBYTE(dword_4F8D80C) = dword_4F8D80C & 0x9F | 0x20;
  qword_4F8D828 = (__int64)"The max size of first-class aggregates in bytes";
  sub_C53130(&qword_4F8D800);
  __cxa_atexit(sub_B2B680, &qword_4F8D800, &qword_4A427C0);
  qword_4F8D720 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8D770 = 0x100000000LL;
  dword_4F8D72C &= 0x8000u;
  word_4F8D730 = 0;
  qword_4F8D768 = (__int64)&unk_4F8D778;
  qword_4F8D738 = 0;
  dword_4F8D728 = v4;
  qword_4F8D740 = 0;
  qword_4F8D748 = 0;
  qword_4F8D750 = 0;
  qword_4F8D758 = 0;
  qword_4F8D760 = 0;
  qword_4F8D780 = 0;
  qword_4F8D788 = (__int64)&unk_4F8D7A0;
  qword_4F8D790 = 1;
  dword_4F8D798 = 0;
  byte_4F8D79C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F8D770;
  if ( (unsigned __int64)(unsigned int)qword_4F8D770 + 1 > HIDWORD(qword_4F8D770) )
  {
    v19 = v5;
    sub_C8D5F0((char *)&unk_4F8D778 - 16, &unk_4F8D778, (unsigned int)qword_4F8D770 + 1LL, 8);
    v6 = (unsigned int)qword_4F8D770;
    v5 = v19;
  }
  *(_QWORD *)(qword_4F8D768 + 8 * v6) = v5;
  qword_4F8D7B0 = (__int64)&unk_49DA090;
  qword_4F8D720 = (__int64)&unk_49DBF90;
  qword_4F8D7C0 = (__int64)&unk_49DC230;
  LODWORD(qword_4F8D770) = qword_4F8D770 + 1;
  qword_4F8D7E0 = (__int64)nullsub_58;
  qword_4F8D7A8 = 0;
  qword_4F8D7D8 = (__int64)sub_B2B5F0;
  qword_4F8D7B8 = 0;
  sub_C53080(&qword_4F8D720, "reg-target-adjust", 17);
  LODWORD(qword_4F8D7A8) = 0;
  BYTE4(qword_4F8D7B8) = 1;
  LODWORD(qword_4F8D7B8) = 0;
  qword_4F8D750 = 55;
  LOBYTE(dword_4F8D72C) = dword_4F8D72C & 0x9F | 0x20;
  qword_4F8D748 = (__int64)"Register target adjustment, range (-10, +10), default 0";
  sub_C53130(&qword_4F8D720);
  __cxa_atexit(sub_B2B680, &qword_4F8D720, &qword_4A427C0);
  qword_4F8D640 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8D690 = 0x100000000LL;
  dword_4F8D64C &= 0x8000u;
  qword_4F8D688 = (__int64)&unk_4F8D698;
  word_4F8D650 = 0;
  qword_4F8D658 = 0;
  dword_4F8D648 = v7;
  qword_4F8D660 = 0;
  qword_4F8D668 = 0;
  qword_4F8D670 = 0;
  qword_4F8D678 = 0;
  qword_4F8D680 = 0;
  qword_4F8D6A0 = 0;
  qword_4F8D6A8 = (__int64)&unk_4F8D6C0;
  qword_4F8D6B0 = 1;
  dword_4F8D6B8 = 0;
  byte_4F8D6BC = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4F8D690;
  if ( (unsigned __int64)(unsigned int)qword_4F8D690 + 1 > HIDWORD(qword_4F8D690) )
  {
    v20 = v8;
    sub_C8D5F0((char *)&unk_4F8D698 - 16, &unk_4F8D698, (unsigned int)qword_4F8D690 + 1LL, 8);
    v9 = (unsigned int)qword_4F8D690;
    v8 = v20;
  }
  *(_QWORD *)(qword_4F8D688 + 8 * v9) = v8;
  qword_4F8D6D0 = (__int64)&unk_49DA090;
  qword_4F8D640 = (__int64)&unk_49DBF90;
  qword_4F8D6E0 = (__int64)&unk_49DC230;
  LODWORD(qword_4F8D690) = qword_4F8D690 + 1;
  qword_4F8D700 = (__int64)nullsub_58;
  qword_4F8D6C8 = 0;
  qword_4F8D6F8 = (__int64)sub_B2B5F0;
  qword_4F8D6D8 = 0;
  sub_C53080(&qword_4F8D640, "pred-target-adjust", 18);
  LODWORD(qword_4F8D6C8) = 0;
  BYTE4(qword_4F8D6D8) = 1;
  LODWORD(qword_4F8D6D8) = 0;
  qword_4F8D670 = 60;
  LOBYTE(dword_4F8D64C) = dword_4F8D64C & 0x9F | 0x20;
  qword_4F8D668 = (__int64)"Predicate reg target adjustment, range (-10, +10), default 0";
  sub_C53130(&qword_4F8D640);
  __cxa_atexit(sub_B2B680, &qword_4F8D640, &qword_4A427C0);
  qword_4F8D560 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8D5B0 = 0x100000000LL;
  dword_4F8D56C &= 0x8000u;
  word_4F8D570 = 0;
  qword_4F8D578 = 0;
  qword_4F8D580 = 0;
  dword_4F8D568 = v10;
  qword_4F8D588 = 0;
  qword_4F8D590 = 0;
  qword_4F8D598 = 0;
  qword_4F8D5A0 = 0;
  qword_4F8D5A8 = (__int64)&unk_4F8D5B8;
  qword_4F8D5C0 = 0;
  qword_4F8D5C8 = (__int64)&unk_4F8D5E0;
  qword_4F8D5D0 = 1;
  dword_4F8D5D8 = 0;
  byte_4F8D5DC = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_4F8D5B0;
  v13 = (unsigned int)qword_4F8D5B0 + 1LL;
  if ( v13 > HIDWORD(qword_4F8D5B0) )
  {
    sub_C8D5F0((char *)&unk_4F8D5B8 - 16, &unk_4F8D5B8, v13, 8);
    v12 = (unsigned int)qword_4F8D5B0;
  }
  *(_QWORD *)(qword_4F8D5A8 + 8 * v12) = v11;
  qword_4F8D5F0 = (__int64)&unk_49D9748;
  qword_4F8D560 = (__int64)&unk_49DC090;
  LODWORD(qword_4F8D5B0) = qword_4F8D5B0 + 1;
  qword_4F8D5E8 = 0;
  qword_4F8D600 = (__int64)&unk_49DC1D0;
  qword_4F8D5F8 = 0;
  qword_4F8D620 = (__int64)nullsub_23;
  qword_4F8D618 = (__int64)sub_984030;
  sub_C53080(&qword_4F8D560, "remat-load-param", 16);
  LOWORD(qword_4F8D5F8) = 257;
  LOBYTE(qword_4F8D5E8) = 1;
  qword_4F8D590 = 63;
  LOBYTE(dword_4F8D56C) = dword_4F8D56C & 0x9F | 0x20;
  qword_4F8D588 = (__int64)"Support remating const ld.param that are not exposed in NVVM IR";
  sub_C53130(&qword_4F8D560);
  __cxa_atexit(sub_984900, &qword_4F8D560, &qword_4A427C0);
  qword_4F8D480 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8D4D0 = 0x100000000LL;
  word_4F8D490 = 0;
  dword_4F8D48C &= 0x8000u;
  qword_4F8D498 = 0;
  qword_4F8D4A0 = 0;
  dword_4F8D488 = v14;
  qword_4F8D4A8 = 0;
  qword_4F8D4B0 = 0;
  qword_4F8D4B8 = 0;
  qword_4F8D4C0 = 0;
  qword_4F8D4C8 = (__int64)&unk_4F8D4D8;
  qword_4F8D4E0 = 0;
  qword_4F8D4E8 = (__int64)&unk_4F8D500;
  qword_4F8D4F0 = 1;
  dword_4F8D4F8 = 0;
  byte_4F8D4FC = 1;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_4F8D4D0;
  v17 = (unsigned int)qword_4F8D4D0 + 1LL;
  if ( v17 > HIDWORD(qword_4F8D4D0) )
  {
    sub_C8D5F0((char *)&unk_4F8D4D8 - 16, &unk_4F8D4D8, v17, 8);
    v16 = (unsigned int)qword_4F8D4D0;
  }
  *(_QWORD *)(qword_4F8D4C8 + 8 * v16) = v15;
  qword_4F8D510 = (__int64)&unk_49D9748;
  qword_4F8D480 = (__int64)&unk_49DC090;
  LODWORD(qword_4F8D4D0) = qword_4F8D4D0 + 1;
  qword_4F8D508 = 0;
  qword_4F8D520 = (__int64)&unk_49DC1D0;
  qword_4F8D518 = 0;
  qword_4F8D540 = (__int64)nullsub_23;
  qword_4F8D538 = (__int64)sub_984030;
  sub_C53080(&qword_4F8D480, "cta-reconfig-aware-rpa", 22);
  LOBYTE(qword_4F8D508) = 1;
  LOWORD(qword_4F8D518) = 257;
  qword_4F8D4B0 = 53;
  LOBYTE(dword_4F8D48C) = dword_4F8D48C & 0x9F | 0x20;
  qword_4F8D4A8 = (__int64)"Enable CTA reconfig aware register pressure analysis.";
  sub_C53130(&qword_4F8D480);
  return __cxa_atexit(sub_984900, &qword_4F8D480, &qword_4A427C0);
}

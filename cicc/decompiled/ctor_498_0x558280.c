// Function: ctor_498
// Address: 0x558280
//
int ctor_498()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // r15
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v15; // [rsp+8h] [rbp-38h]
  __int64 v16; // [rsp+8h] [rbp-38h]

  qword_50098E0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5009930 = 0x100000000LL;
  dword_50098EC &= 0x8000u;
  word_50098F0 = 0;
  qword_50098F8 = 0;
  qword_5009900 = 0;
  dword_50098E8 = v0;
  qword_5009908 = 0;
  qword_5009910 = 0;
  qword_5009918 = 0;
  qword_5009920 = 0;
  qword_5009928 = (__int64)&unk_5009938;
  qword_5009940 = 0;
  qword_5009948 = (__int64)&unk_5009960;
  qword_5009950 = 1;
  dword_5009958 = 0;
  byte_500995C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5009930;
  v3 = (unsigned int)qword_5009930 + 1LL;
  if ( v3 > HIDWORD(qword_5009930) )
  {
    sub_C8D5F0((char *)&unk_5009938 - 16, &unk_5009938, v3, 8);
    v2 = (unsigned int)qword_5009930;
  }
  *(_QWORD *)(qword_5009928 + 8 * v2) = v1;
  qword_5009970 = (__int64)&unk_49D9748;
  qword_50098E0 = (__int64)&unk_49DC090;
  qword_5009980 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5009930) = qword_5009930 + 1;
  qword_50099A0 = (__int64)nullsub_23;
  qword_5009968 = 0;
  qword_5009998 = (__int64)sub_984030;
  qword_5009978 = 0;
  sub_C53080(&qword_50098E0, "norm-preserve-order", 19);
  LOWORD(qword_5009978) = 256;
  LOBYTE(qword_5009968) = 0;
  qword_5009910 = 36;
  LOBYTE(dword_50098EC) = dword_50098EC & 0x9F | 0x20;
  qword_5009908 = (__int64)"Preserves original instruction order";
  sub_C53130(&qword_50098E0);
  __cxa_atexit(sub_984900, &qword_50098E0, &qword_4A427C0);
  qword_5009800 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5009850 = 0x100000000LL;
  dword_500980C &= 0x8000u;
  qword_5009848 = (__int64)&unk_5009858;
  word_5009810 = 0;
  qword_5009818 = 0;
  dword_5009808 = v4;
  qword_5009820 = 0;
  qword_5009828 = 0;
  qword_5009830 = 0;
  qword_5009838 = 0;
  qword_5009840 = 0;
  qword_5009860 = 0;
  qword_5009868 = (__int64)&unk_5009880;
  qword_5009870 = 1;
  dword_5009878 = 0;
  byte_500987C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5009850;
  if ( (unsigned __int64)(unsigned int)qword_5009850 + 1 > HIDWORD(qword_5009850) )
  {
    v15 = v5;
    sub_C8D5F0((char *)&unk_5009858 - 16, &unk_5009858, (unsigned int)qword_5009850 + 1LL, 8);
    v6 = (unsigned int)qword_5009850;
    v5 = v15;
  }
  *(_QWORD *)(qword_5009848 + 8 * v6) = v5;
  qword_5009890 = (__int64)&unk_49D9748;
  qword_5009800 = (__int64)&unk_49DC090;
  qword_50098A0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5009850) = qword_5009850 + 1;
  qword_50098C0 = (__int64)nullsub_23;
  qword_5009888 = 0;
  qword_50098B8 = (__int64)sub_984030;
  qword_5009898 = 0;
  sub_C53080(&qword_5009800, "norm-rename-all", 15);
  LOWORD(qword_5009898) = 257;
  LOBYTE(qword_5009888) = 1;
  qword_5009830 = 47;
  LOBYTE(dword_500980C) = dword_500980C & 0x9F | 0x20;
  qword_5009828 = (__int64)"Renames all instructions (including user-named)";
  sub_C53130(&qword_5009800);
  __cxa_atexit(sub_984900, &qword_5009800, &qword_4A427C0);
  qword_5009720 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5009770 = 0x100000000LL;
  dword_500972C &= 0x8000u;
  qword_5009768 = (__int64)&unk_5009778;
  word_5009730 = 0;
  qword_5009738 = 0;
  dword_5009728 = v7;
  qword_5009740 = 0;
  qword_5009748 = 0;
  qword_5009750 = 0;
  qword_5009758 = 0;
  qword_5009760 = 0;
  qword_5009780 = 0;
  qword_5009788 = (__int64)&unk_50097A0;
  qword_5009790 = 1;
  dword_5009798 = 0;
  byte_500979C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_5009770;
  if ( (unsigned __int64)(unsigned int)qword_5009770 + 1 > HIDWORD(qword_5009770) )
  {
    v16 = v8;
    sub_C8D5F0((char *)&unk_5009778 - 16, &unk_5009778, (unsigned int)qword_5009770 + 1LL, 8);
    v9 = (unsigned int)qword_5009770;
    v8 = v16;
  }
  *(_QWORD *)(qword_5009768 + 8 * v9) = v8;
  qword_50097B0 = (__int64)&unk_49D9748;
  qword_5009720 = (__int64)&unk_49DC090;
  qword_50097C0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5009770) = qword_5009770 + 1;
  qword_50097E0 = (__int64)nullsub_23;
  qword_50097A8 = 0;
  qword_50097D8 = (__int64)sub_984030;
  qword_50097B8 = 0;
  sub_C53080(&qword_5009720, "norm-fold-all", 13);
  LOWORD(qword_50097B8) = 257;
  LOBYTE(qword_50097A8) = 1;
  qword_5009750 = 54;
  LOBYTE(dword_500972C) = dword_500972C & 0x9F | 0x20;
  qword_5009748 = (__int64)"Folds all regular instructions (including pre-outputs)";
  sub_C53130(&qword_5009720);
  __cxa_atexit(sub_984900, &qword_5009720, &qword_4A427C0);
  qword_5009640 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5009690 = 0x100000000LL;
  dword_500964C &= 0x8000u;
  word_5009650 = 0;
  qword_5009688 = (__int64)&unk_5009698;
  qword_5009658 = 0;
  dword_5009648 = v10;
  qword_5009660 = 0;
  qword_5009668 = 0;
  qword_5009670 = 0;
  qword_5009678 = 0;
  qword_5009680 = 0;
  qword_50096A0 = 0;
  qword_50096A8 = (__int64)&unk_50096C0;
  qword_50096B0 = 1;
  dword_50096B8 = 0;
  byte_50096BC = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5009690;
  v13 = (unsigned int)qword_5009690 + 1LL;
  if ( v13 > HIDWORD(qword_5009690) )
  {
    sub_C8D5F0((char *)&unk_5009698 - 16, &unk_5009698, v13, 8);
    v12 = (unsigned int)qword_5009690;
  }
  *(_QWORD *)(qword_5009688 + 8 * v12) = v11;
  qword_50096D0 = (__int64)&unk_49D9748;
  qword_5009640 = (__int64)&unk_49DC090;
  qword_50096E0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5009690) = qword_5009690 + 1;
  qword_5009700 = (__int64)nullsub_23;
  qword_50096C8 = 0;
  qword_50096F8 = (__int64)sub_984030;
  qword_50096D8 = 0;
  sub_C53080(&qword_5009640, "norm-reorder-operands", 21);
  LOBYTE(qword_50096C8) = 1;
  qword_5009670 = 55;
  LOBYTE(dword_500964C) = dword_500964C & 0x9F | 0x20;
  LOWORD(qword_50096D8) = 257;
  qword_5009668 = (__int64)"Sorts and reorders operands in commutative instructions";
  sub_C53130(&qword_5009640);
  return __cxa_atexit(sub_984900, &qword_5009640, &qword_4A427C0);
}

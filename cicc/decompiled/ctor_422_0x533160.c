// Function: ctor_422
// Address: 0x533160
//
int ctor_422()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-38h]

  qword_4FF17A0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF17F0 = 0x100000000LL;
  dword_4FF17AC &= 0x8000u;
  word_4FF17B0 = 0;
  qword_4FF17B8 = 0;
  qword_4FF17C0 = 0;
  dword_4FF17A8 = v0;
  qword_4FF17C8 = 0;
  qword_4FF17D0 = 0;
  qword_4FF17D8 = 0;
  qword_4FF17E0 = 0;
  qword_4FF17E8 = (__int64)&unk_4FF17F8;
  qword_4FF1800 = 0;
  qword_4FF1808 = (__int64)&unk_4FF1820;
  qword_4FF1810 = 1;
  dword_4FF1818 = 0;
  byte_4FF181C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FF17F0;
  v3 = (unsigned int)qword_4FF17F0 + 1LL;
  if ( v3 > HIDWORD(qword_4FF17F0) )
  {
    sub_C8D5F0((char *)&unk_4FF17F8 - 16, &unk_4FF17F8, v3, 8);
    v2 = (unsigned int)qword_4FF17F0;
  }
  *(_QWORD *)(qword_4FF17E8 + 8 * v2) = v1;
  qword_4FF1830 = (__int64)&unk_49D9748;
  qword_4FF17A0 = (__int64)&unk_49DC090;
  LODWORD(qword_4FF17F0) = qword_4FF17F0 + 1;
  qword_4FF1828 = 0;
  qword_4FF1840 = (__int64)&unk_49DC1D0;
  qword_4FF1838 = 0;
  qword_4FF1860 = (__int64)nullsub_23;
  qword_4FF1858 = (__int64)sub_984030;
  sub_C53080(&qword_4FF17A0, "optimize-non-fmv-callers", 24);
  qword_4FF17C8 = (__int64)"Statically resolve calls to versioned functions from non-versioned callers.";
  LOWORD(qword_4FF1838) = 257;
  LOBYTE(qword_4FF1828) = 1;
  qword_4FF17D0 = 75;
  LOBYTE(dword_4FF17AC) = dword_4FF17AC & 0x9F | 0x20;
  sub_C53130(&qword_4FF17A0);
  __cxa_atexit(sub_984900, &qword_4FF17A0, &qword_4A427C0);
  qword_4FF16C0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF1710 = 0x100000000LL;
  dword_4FF16CC &= 0x8000u;
  word_4FF16D0 = 0;
  qword_4FF16D8 = 0;
  qword_4FF16E0 = 0;
  dword_4FF16C8 = v4;
  qword_4FF16E8 = 0;
  qword_4FF16F0 = 0;
  qword_4FF16F8 = 0;
  qword_4FF1700 = 0;
  qword_4FF1708 = (__int64)&unk_4FF1718;
  qword_4FF1720 = 0;
  qword_4FF1728 = (__int64)&unk_4FF1740;
  qword_4FF1730 = 1;
  dword_4FF1738 = 0;
  byte_4FF173C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FF1710;
  if ( (unsigned __int64)(unsigned int)qword_4FF1710 + 1 > HIDWORD(qword_4FF1710) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_4FF1718 - 16, &unk_4FF1718, (unsigned int)qword_4FF1710 + 1LL, 8);
    v6 = (unsigned int)qword_4FF1710;
    v5 = v12;
  }
  *(_QWORD *)(qword_4FF1708 + 8 * v6) = v5;
  qword_4FF1750 = (__int64)&unk_49D9748;
  qword_4FF16C0 = (__int64)&unk_49DC090;
  LODWORD(qword_4FF1710) = qword_4FF1710 + 1;
  qword_4FF1748 = 0;
  qword_4FF1760 = (__int64)&unk_49DC1D0;
  qword_4FF1758 = 0;
  qword_4FF1780 = (__int64)nullsub_23;
  qword_4FF1778 = (__int64)sub_984030;
  sub_C53080(&qword_4FF16C0, "enable-coldcc-stress-test", 25);
  qword_4FF16F0 = 78;
  qword_4FF16E8 = (__int64)"Enable stress test of coldcc by adding calling conv to all internal functions.";
  LOWORD(qword_4FF1758) = 256;
  LOBYTE(qword_4FF1748) = 0;
  LOBYTE(dword_4FF16CC) = dword_4FF16CC & 0x9F | 0x20;
  sub_C53130(&qword_4FF16C0);
  __cxa_atexit(sub_984900, &qword_4FF16C0, &qword_4A427C0);
  qword_4FF15E0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FF165C = 1;
  qword_4FF1630 = 0x100000000LL;
  dword_4FF15EC &= 0x8000u;
  qword_4FF15F8 = 0;
  qword_4FF1600 = 0;
  qword_4FF1608 = 0;
  dword_4FF15E8 = v7;
  word_4FF15F0 = 0;
  qword_4FF1610 = 0;
  qword_4FF1618 = 0;
  qword_4FF1620 = 0;
  qword_4FF1628 = (__int64)&unk_4FF1638;
  qword_4FF1640 = 0;
  qword_4FF1648 = (__int64)&unk_4FF1660;
  qword_4FF1650 = 1;
  dword_4FF1658 = 0;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4FF1630;
  v10 = (unsigned int)qword_4FF1630 + 1LL;
  if ( v10 > HIDWORD(qword_4FF1630) )
  {
    sub_C8D5F0((char *)&unk_4FF1638 - 16, &unk_4FF1638, v10, 8);
    v9 = (unsigned int)qword_4FF1630;
  }
  *(_QWORD *)(qword_4FF1628 + 8 * v9) = v8;
  LODWORD(qword_4FF1630) = qword_4FF1630 + 1;
  qword_4FF1668 = 0;
  qword_4FF1670 = (__int64)&unk_49DA090;
  qword_4FF1678 = 0;
  qword_4FF15E0 = (__int64)&unk_49DBF90;
  qword_4FF1680 = (__int64)&unk_49DC230;
  qword_4FF16A0 = (__int64)nullsub_58;
  qword_4FF1698 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FF15E0, "coldcc-rel-freq", 15);
  LODWORD(qword_4FF1668) = 2;
  BYTE4(qword_4FF1678) = 1;
  LODWORD(qword_4FF1678) = 2;
  qword_4FF1610 = 137;
  LOBYTE(dword_4FF15EC) = dword_4FF15EC & 0x9F | 0x20;
  qword_4FF1608 = (__int64)"Maximum block frequency, expressed as a percentage of caller's entry frequency, for a call si"
                           "te to be considered cold for enabling coldcc";
  sub_C53130(&qword_4FF15E0);
  return __cxa_atexit(sub_B2B680, &qword_4FF15E0, &qword_4A427C0);
}

// Function: ctor_382
// Address: 0x51a300
//
int ctor_382()
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

  qword_4FDB880 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FDB8D0 = 0x100000000LL;
  dword_4FDB88C &= 0x8000u;
  word_4FDB890 = 0;
  qword_4FDB898 = 0;
  qword_4FDB8A0 = 0;
  dword_4FDB888 = v0;
  qword_4FDB8A8 = 0;
  qword_4FDB8B0 = 0;
  qword_4FDB8B8 = 0;
  qword_4FDB8C0 = 0;
  qword_4FDB8C8 = (__int64)&unk_4FDB8D8;
  qword_4FDB8E0 = 0;
  qword_4FDB8E8 = (__int64)&unk_4FDB900;
  qword_4FDB8F0 = 1;
  dword_4FDB8F8 = 0;
  byte_4FDB8FC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FDB8D0;
  v3 = (unsigned int)qword_4FDB8D0 + 1LL;
  if ( v3 > HIDWORD(qword_4FDB8D0) )
  {
    sub_C8D5F0((char *)&unk_4FDB8D8 - 16, &unk_4FDB8D8, v3, 8);
    v2 = (unsigned int)qword_4FDB8D0;
  }
  *(_QWORD *)(qword_4FDB8C8 + 8 * v2) = v1;
  qword_4FDB910 = (__int64)&unk_49D9748;
  qword_4FDB880 = (__int64)&unk_49DC090;
  qword_4FDB920 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FDB8D0) = qword_4FDB8D0 + 1;
  qword_4FDB940 = (__int64)nullsub_23;
  qword_4FDB908 = 0;
  qword_4FDB938 = (__int64)sub_984030;
  qword_4FDB918 = 0;
  sub_C53080(&qword_4FDB880, "iv-users-check-mul", 18);
  qword_4FDB8A8 = (__int64)"Check if SCEV MULExpr can be considered as an IV";
  LOWORD(qword_4FDB918) = 257;
  LOBYTE(qword_4FDB908) = 1;
  qword_4FDB8B0 = 48;
  LOBYTE(dword_4FDB88C) = dword_4FDB88C & 0x9F | 0x40;
  sub_C53130(&qword_4FDB880);
  __cxa_atexit(sub_984900, &qword_4FDB880, &qword_4A427C0);
  qword_4FDB7A0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FDB7F0 = 0x100000000LL;
  dword_4FDB7AC &= 0x8000u;
  qword_4FDB7E8 = (__int64)&unk_4FDB7F8;
  word_4FDB7B0 = 0;
  qword_4FDB7B8 = 0;
  dword_4FDB7A8 = v4;
  qword_4FDB7C0 = 0;
  qword_4FDB7C8 = 0;
  qword_4FDB7D0 = 0;
  qword_4FDB7D8 = 0;
  qword_4FDB7E0 = 0;
  qword_4FDB800 = 0;
  qword_4FDB808 = (__int64)&unk_4FDB820;
  qword_4FDB810 = 1;
  dword_4FDB818 = 0;
  byte_4FDB81C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FDB7F0;
  if ( (unsigned __int64)(unsigned int)qword_4FDB7F0 + 1 > HIDWORD(qword_4FDB7F0) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_4FDB7F8 - 16, &unk_4FDB7F8, (unsigned int)qword_4FDB7F0 + 1LL, 8);
    v6 = (unsigned int)qword_4FDB7F0;
    v5 = v12;
  }
  *(_QWORD *)(qword_4FDB7E8 + 8 * v6) = v5;
  qword_4FDB830 = (__int64)&unk_49D9748;
  qword_4FDB7A0 = (__int64)&unk_49DC090;
  qword_4FDB840 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FDB7F0) = qword_4FDB7F0 + 1;
  qword_4FDB860 = (__int64)nullsub_23;
  qword_4FDB828 = 0;
  qword_4FDB858 = (__int64)sub_984030;
  qword_4FDB838 = 0;
  sub_C53080(&qword_4FDB7A0, "check-sxtopt", 12);
  qword_4FDB7C8 = (__int64)"Check if sign extension can be eliminated";
  LOWORD(qword_4FDB838) = 257;
  LOBYTE(qword_4FDB828) = 1;
  qword_4FDB7D0 = 41;
  LOBYTE(dword_4FDB7AC) = dword_4FDB7AC & 0x9F | 0x20;
  sub_C53130(&qword_4FDB7A0);
  __cxa_atexit(sub_984900, &qword_4FDB7A0, &qword_4A427C0);
  qword_4FDB6C0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FDB710 = 0x100000000LL;
  dword_4FDB6CC &= 0x8000u;
  word_4FDB6D0 = 0;
  qword_4FDB708 = (__int64)&unk_4FDB718;
  qword_4FDB6D8 = 0;
  dword_4FDB6C8 = v7;
  qword_4FDB6E0 = 0;
  qword_4FDB6E8 = 0;
  qword_4FDB6F0 = 0;
  qword_4FDB6F8 = 0;
  qword_4FDB700 = 0;
  qword_4FDB720 = 0;
  qword_4FDB728 = (__int64)&unk_4FDB740;
  qword_4FDB730 = 1;
  dword_4FDB738 = 0;
  byte_4FDB73C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4FDB710;
  v10 = (unsigned int)qword_4FDB710 + 1LL;
  if ( v10 > HIDWORD(qword_4FDB710) )
  {
    sub_C8D5F0((char *)&unk_4FDB718 - 16, &unk_4FDB718, v10, 8);
    v9 = (unsigned int)qword_4FDB710;
  }
  *(_QWORD *)(qword_4FDB708 + 8 * v9) = v8;
  qword_4FDB750 = (__int64)&unk_49D9748;
  qword_4FDB6C0 = (__int64)&unk_49DC090;
  qword_4FDB760 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FDB710) = qword_4FDB710 + 1;
  qword_4FDB780 = (__int64)nullsub_23;
  qword_4FDB748 = 0;
  qword_4FDB778 = (__int64)sub_984030;
  qword_4FDB758 = 0;
  sub_C53080(&qword_4FDB6C0, "iv-skip-sxt", 11);
  LOBYTE(qword_4FDB748) = 0;
  LOWORD(qword_4FDB758) = 256;
  qword_4FDB6E8 = (__int64)"Ignore SignExtendedExpr for IV";
  qword_4FDB6F0 = 30;
  LOBYTE(dword_4FDB6CC) = dword_4FDB6CC & 0x9F | 0x20;
  sub_C53130(&qword_4FDB6C0);
  return __cxa_atexit(sub_984900, &qword_4FDB6C0, &qword_4A427C0);
}

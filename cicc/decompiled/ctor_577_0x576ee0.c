// Function: ctor_577
// Address: 0x576ee0
//
int ctor_577()
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
  __int64 v11; // rax
  __int64 v12; // rdx
  int v13; // edx
  __int64 v14; // r12
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  int v17; // edx
  __int64 v18; // rbx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v22; // [rsp+8h] [rbp-38h]
  __int64 v23; // [rsp+8h] [rbp-38h]
  __int64 v24; // [rsp+8h] [rbp-38h]

  qword_5022B60 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5022BB0 = 0x100000000LL;
  dword_5022B6C &= 0x8000u;
  word_5022B70 = 0;
  qword_5022B78 = 0;
  qword_5022B80 = 0;
  dword_5022B68 = v0;
  qword_5022B88 = 0;
  qword_5022B90 = 0;
  qword_5022B98 = 0;
  qword_5022BA0 = 0;
  qword_5022BA8 = (__int64)&unk_5022BB8;
  qword_5022BC0 = 0;
  qword_5022BC8 = (__int64)&unk_5022BE0;
  qword_5022BD0 = 1;
  dword_5022BD8 = 0;
  byte_5022BDC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5022BB0;
  v3 = (unsigned int)qword_5022BB0 + 1LL;
  if ( v3 > HIDWORD(qword_5022BB0) )
  {
    sub_C8D5F0((char *)&unk_5022BB8 - 16, &unk_5022BB8, v3, 8);
    v2 = (unsigned int)qword_5022BB0;
  }
  *(_QWORD *)(qword_5022BA8 + 8 * v2) = v1;
  qword_5022BF0 = (__int64)&unk_49D9748;
  qword_5022B60 = (__int64)&unk_49DC090;
  qword_5022C00 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5022BB0) = qword_5022BB0 + 1;
  qword_5022C20 = (__int64)nullsub_23;
  qword_5022BE8 = 0;
  qword_5022C18 = (__int64)sub_984030;
  qword_5022BF8 = 0;
  sub_C53080(&qword_5022B60, "aggressive-ext-opt", 18);
  qword_5022B90 = 33;
  LOBYTE(dword_5022B6C) = dword_5022B6C & 0x9F | 0x20;
  qword_5022B88 = (__int64)"Aggressive extension optimization";
  sub_C53130(&qword_5022B60);
  __cxa_atexit(sub_984900, &qword_5022B60, &qword_4A427C0);
  qword_5022A80 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5022AD0 = 0x100000000LL;
  dword_5022A8C &= 0x8000u;
  qword_5022AC8 = (__int64)&unk_5022AD8;
  word_5022A90 = 0;
  qword_5022A98 = 0;
  dword_5022A88 = v4;
  qword_5022AA0 = 0;
  qword_5022AA8 = 0;
  qword_5022AB0 = 0;
  qword_5022AB8 = 0;
  qword_5022AC0 = 0;
  qword_5022AE0 = 0;
  qword_5022AE8 = (__int64)&unk_5022B00;
  qword_5022AF0 = 1;
  dword_5022AF8 = 0;
  byte_5022AFC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5022AD0;
  if ( (unsigned __int64)(unsigned int)qword_5022AD0 + 1 > HIDWORD(qword_5022AD0) )
  {
    v22 = v5;
    sub_C8D5F0((char *)&unk_5022AD8 - 16, &unk_5022AD8, (unsigned int)qword_5022AD0 + 1LL, 8);
    v6 = (unsigned int)qword_5022AD0;
    v5 = v22;
  }
  *(_QWORD *)(qword_5022AC8 + 8 * v6) = v5;
  qword_5022B10 = (__int64)&unk_49D9748;
  qword_5022A80 = (__int64)&unk_49DC090;
  qword_5022B20 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5022AD0) = qword_5022AD0 + 1;
  qword_5022B40 = (__int64)nullsub_23;
  qword_5022B08 = 0;
  qword_5022B38 = (__int64)sub_984030;
  qword_5022B18 = 0;
  sub_C53080(&qword_5022A80, "disable-peephole", 16);
  LOWORD(qword_5022B18) = 256;
  LOBYTE(qword_5022B08) = 0;
  qword_5022AB0 = 30;
  LOBYTE(dword_5022A8C) = dword_5022A8C & 0x9F | 0x20;
  qword_5022AA8 = (__int64)"Disable the peephole optimizer";
  sub_C53130(&qword_5022A80);
  __cxa_atexit(sub_984900, &qword_5022A80, &qword_4A427C0);
  qword_50229A0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50229F0 = 0x100000000LL;
  dword_50229AC &= 0x8000u;
  qword_50229E8 = (__int64)&unk_50229F8;
  word_50229B0 = 0;
  qword_50229B8 = 0;
  dword_50229A8 = v7;
  qword_50229C0 = 0;
  qword_50229C8 = 0;
  qword_50229D0 = 0;
  qword_50229D8 = 0;
  qword_50229E0 = 0;
  qword_5022A00 = 0;
  qword_5022A08 = (__int64)&unk_5022A20;
  qword_5022A10 = 1;
  dword_5022A18 = 0;
  byte_5022A1C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_50229F0;
  if ( (unsigned __int64)(unsigned int)qword_50229F0 + 1 > HIDWORD(qword_50229F0) )
  {
    v23 = v8;
    sub_C8D5F0((char *)&unk_50229F8 - 16, &unk_50229F8, (unsigned int)qword_50229F0 + 1LL, 8);
    v9 = (unsigned int)qword_50229F0;
    v8 = v23;
  }
  *(_QWORD *)(qword_50229E8 + 8 * v9) = v8;
  qword_5022A30 = (__int64)&unk_49D9748;
  qword_50229A0 = (__int64)&unk_49DC090;
  qword_5022A40 = (__int64)&unk_49DC1D0;
  LODWORD(qword_50229F0) = qword_50229F0 + 1;
  qword_5022A60 = (__int64)nullsub_23;
  qword_5022A28 = 0;
  qword_5022A58 = (__int64)sub_984030;
  qword_5022A38 = 0;
  sub_C53080(&qword_50229A0, "disable-adv-copy-opt", 20);
  LOWORD(qword_5022A38) = 256;
  LOBYTE(qword_5022A28) = 0;
  qword_50229D0 = 34;
  LOBYTE(dword_50229AC) = dword_50229AC & 0x9F | 0x20;
  qword_50229C8 = (__int64)"Disable advanced copy optimization";
  sub_C53130(&qword_50229A0);
  __cxa_atexit(sub_984900, &qword_50229A0, &qword_4A427C0);
  qword_50228C0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5022910 = 0x100000000LL;
  dword_50228CC &= 0x8000u;
  qword_5022908 = (__int64)&unk_5022918;
  word_50228D0 = 0;
  qword_50228D8 = 0;
  dword_50228C8 = v10;
  qword_50228E0 = 0;
  qword_50228E8 = 0;
  qword_50228F0 = 0;
  qword_50228F8 = 0;
  qword_5022900 = 0;
  qword_5022920 = 0;
  qword_5022928 = (__int64)&unk_5022940;
  qword_5022930 = 1;
  dword_5022938 = 0;
  byte_502293C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5022910;
  if ( (unsigned __int64)(unsigned int)qword_5022910 + 1 > HIDWORD(qword_5022910) )
  {
    v24 = v11;
    sub_C8D5F0((char *)&unk_5022918 - 16, &unk_5022918, (unsigned int)qword_5022910 + 1LL, 8);
    v12 = (unsigned int)qword_5022910;
    v11 = v24;
  }
  *(_QWORD *)(qword_5022908 + 8 * v12) = v11;
  qword_5022950 = (__int64)&unk_49D9748;
  qword_50228C0 = (__int64)&unk_49DC090;
  qword_5022960 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5022910) = qword_5022910 + 1;
  qword_5022980 = (__int64)nullsub_23;
  qword_5022948 = 0;
  qword_5022978 = (__int64)sub_984030;
  qword_5022958 = 0;
  sub_C53080(&qword_50228C0, "disable-non-allocatable-phys-copy-opt", 37);
  LOWORD(qword_5022958) = 256;
  LOBYTE(qword_5022948) = 0;
  qword_50228F0 = 59;
  LOBYTE(dword_50228CC) = dword_50228CC & 0x9F | 0x20;
  qword_50228E8 = (__int64)"Disable non-allocatable physical register copy optimization";
  sub_C53130(&qword_50228C0);
  __cxa_atexit(sub_984900, &qword_50228C0, &qword_4A427C0);
  qword_50227E0 = (__int64)&unk_49DC150;
  v13 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5022830 = 0x100000000LL;
  word_50227F0 = 0;
  dword_50227EC &= 0x8000u;
  qword_50227F8 = 0;
  qword_5022800 = 0;
  dword_50227E8 = v13;
  qword_5022808 = 0;
  qword_5022810 = 0;
  qword_5022818 = 0;
  qword_5022820 = 0;
  qword_5022828 = (__int64)&unk_5022838;
  qword_5022840 = 0;
  qword_5022848 = (__int64)&unk_5022860;
  qword_5022850 = 1;
  dword_5022858 = 0;
  byte_502285C = 1;
  v14 = sub_C57470();
  v15 = (unsigned int)qword_5022830;
  v16 = (unsigned int)qword_5022830 + 1LL;
  if ( v16 > HIDWORD(qword_5022830) )
  {
    sub_C8D5F0((char *)&unk_5022838 - 16, &unk_5022838, v16, 8);
    v15 = (unsigned int)qword_5022830;
  }
  *(_QWORD *)(qword_5022828 + 8 * v15) = v14;
  qword_5022870 = (__int64)&unk_49D9728;
  qword_50227E0 = (__int64)&unk_49DBF10;
  LODWORD(qword_5022830) = qword_5022830 + 1;
  qword_5022868 = 0;
  qword_5022880 = (__int64)&unk_49DC290;
  qword_5022878 = 0;
  qword_50228A0 = (__int64)nullsub_24;
  qword_5022898 = (__int64)sub_984050;
  sub_C53080(&qword_50227E0, "rewrite-phi-limit", 17);
  LODWORD(qword_5022868) = 10;
  BYTE4(qword_5022878) = 1;
  LODWORD(qword_5022878) = 10;
  qword_5022810 = 40;
  LOBYTE(dword_50227EC) = dword_50227EC & 0x9F | 0x20;
  qword_5022808 = (__int64)"Limit the length of PHI chains to lookup";
  sub_C53130(&qword_50227E0);
  __cxa_atexit(sub_984970, &qword_50227E0, &qword_4A427C0);
  qword_5022700 = (__int64)&unk_49DC150;
  v17 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_502270C &= 0x8000u;
  word_5022710 = 0;
  qword_5022750 = 0x100000000LL;
  qword_5022718 = 0;
  qword_5022720 = 0;
  qword_5022728 = 0;
  dword_5022708 = v17;
  qword_5022730 = 0;
  qword_5022738 = 0;
  qword_5022740 = 0;
  qword_5022748 = (__int64)&unk_5022758;
  qword_5022760 = 0;
  qword_5022768 = (__int64)&unk_5022780;
  qword_5022770 = 1;
  dword_5022778 = 0;
  byte_502277C = 1;
  v18 = sub_C57470();
  v19 = (unsigned int)qword_5022750;
  v20 = (unsigned int)qword_5022750 + 1LL;
  if ( v20 > HIDWORD(qword_5022750) )
  {
    sub_C8D5F0((char *)&unk_5022758 - 16, &unk_5022758, v20, 8);
    v19 = (unsigned int)qword_5022750;
  }
  *(_QWORD *)(qword_5022748 + 8 * v19) = v18;
  qword_5022790 = (__int64)&unk_49D9728;
  qword_5022700 = (__int64)&unk_49DBF10;
  LODWORD(qword_5022750) = qword_5022750 + 1;
  qword_5022788 = 0;
  qword_50227A0 = (__int64)&unk_49DC290;
  qword_5022798 = 0;
  qword_50227C0 = (__int64)nullsub_24;
  qword_50227B8 = (__int64)sub_984050;
  sub_C53080(&qword_5022700, "recurrence-chain-limit", 22);
  LODWORD(qword_5022788) = 3;
  BYTE4(qword_5022798) = 1;
  LODWORD(qword_5022798) = 3;
  qword_5022730 = 84;
  LOBYTE(dword_502270C) = dword_502270C & 0x9F | 0x20;
  qword_5022728 = (__int64)"Maximum length of recurrence chain when evaluating the benefit of commuting operands";
  sub_C53130(&qword_5022700);
  return __cxa_atexit(sub_984970, &qword_5022700, &qword_4A427C0);
}

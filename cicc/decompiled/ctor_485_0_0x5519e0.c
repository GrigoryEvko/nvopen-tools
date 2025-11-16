// Function: ctor_485_0
// Address: 0x5519e0
//
int ctor_485_0()
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
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // edx
  __int64 v18; // rax
  __int64 v19; // rdx
  int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rdx
  int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rdx
  int v26; // edx
  __int64 v27; // rbx
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  __int64 v31; // [rsp+8h] [rbp-38h]
  __int64 v32; // [rsp+8h] [rbp-38h]
  __int64 v33; // [rsp+8h] [rbp-38h]
  __int64 v34; // [rsp+8h] [rbp-38h]
  __int64 v35; // [rsp+8h] [rbp-38h]
  __int64 v36; // [rsp+8h] [rbp-38h]

  qword_5006D20 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5006D70 = 0x100000000LL;
  dword_5006D2C &= 0x8000u;
  word_5006D30 = 0;
  qword_5006D38 = 0;
  qword_5006D40 = 0;
  dword_5006D28 = v0;
  qword_5006D48 = 0;
  qword_5006D50 = 0;
  qword_5006D58 = 0;
  qword_5006D60 = 0;
  qword_5006D68 = (__int64)&unk_5006D78;
  qword_5006D80 = 0;
  qword_5006D88 = (__int64)&unk_5006DA0;
  qword_5006D90 = 1;
  dword_5006D98 = 0;
  byte_5006D9C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5006D70;
  v3 = (unsigned int)qword_5006D70 + 1LL;
  if ( v3 > HIDWORD(qword_5006D70) )
  {
    sub_C8D5F0((char *)&unk_5006D78 - 16, &unk_5006D78, v3, 8);
    v2 = (unsigned int)qword_5006D70;
  }
  *(_QWORD *)(qword_5006D68 + 8 * v2) = v1;
  LODWORD(qword_5006D70) = qword_5006D70 + 1;
  qword_5006DA8 = 0;
  qword_5006DB0 = (__int64)&unk_49D9728;
  qword_5006DB8 = 0;
  qword_5006D20 = (__int64)&unk_49DBF10;
  qword_5006DC0 = (__int64)&unk_49DC290;
  qword_5006DE0 = (__int64)nullsub_24;
  qword_5006DD8 = (__int64)sub_984050;
  sub_C53080(&qword_5006D20, "bonus-inst-threshold", 20);
  LODWORD(qword_5006DA8) = 1;
  BYTE4(qword_5006DB8) = 1;
  LODWORD(qword_5006DB8) = 1;
  qword_5006D50 = 54;
  LOBYTE(dword_5006D2C) = dword_5006D2C & 0x9F | 0x20;
  qword_5006D48 = (__int64)"Control the number of bonus instructions (default = 1)";
  sub_C53130(&qword_5006D20);
  __cxa_atexit(sub_984970, &qword_5006D20, &qword_4A427C0);
  qword_5006C40 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5006C90 = 0x100000000LL;
  word_5006C50 = 0;
  dword_5006C4C &= 0x8000u;
  qword_5006C58 = 0;
  qword_5006C60 = 0;
  dword_5006C48 = v4;
  qword_5006C68 = 0;
  qword_5006C70 = 0;
  qword_5006C78 = 0;
  qword_5006C80 = 0;
  qword_5006C88 = (__int64)&unk_5006C98;
  qword_5006CA0 = 0;
  qword_5006CA8 = (__int64)&unk_5006CC0;
  qword_5006CB0 = 1;
  dword_5006CB8 = 0;
  byte_5006CBC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5006C90;
  v7 = (unsigned int)qword_5006C90 + 1LL;
  if ( v7 > HIDWORD(qword_5006C90) )
  {
    sub_C8D5F0((char *)&unk_5006C98 - 16, &unk_5006C98, v7, 8);
    v6 = (unsigned int)qword_5006C90;
  }
  *(_QWORD *)(qword_5006C88 + 8 * v6) = v5;
  qword_5006CD0 = (__int64)&unk_49D9748;
  qword_5006C40 = (__int64)&unk_49DC090;
  qword_5006CE0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5006C90) = qword_5006C90 + 1;
  qword_5006D00 = (__int64)nullsub_23;
  qword_5006CC8 = 0;
  qword_5006CF8 = (__int64)sub_984030;
  qword_5006CD8 = 0;
  sub_C53080(&qword_5006C40, "keep-loops", 10);
  LOBYTE(qword_5006CC8) = 1;
  qword_5006C70 = 50;
  LOBYTE(dword_5006C4C) = dword_5006C4C & 0x9F | 0x20;
  LOWORD(qword_5006CD8) = 257;
  qword_5006C68 = (__int64)"Preserve canonical loop structure (default = true)";
  sub_C53130(&qword_5006C40);
  __cxa_atexit(sub_984900, &qword_5006C40, &qword_4A427C0);
  qword_5006B60 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5006BDC = 1;
  qword_5006BB0 = 0x100000000LL;
  dword_5006B6C &= 0x8000u;
  qword_5006BA8 = (__int64)&unk_5006BB8;
  qword_5006B78 = 0;
  qword_5006B80 = 0;
  dword_5006B68 = v8;
  word_5006B70 = 0;
  qword_5006B88 = 0;
  qword_5006B90 = 0;
  qword_5006B98 = 0;
  qword_5006BA0 = 0;
  qword_5006BC0 = 0;
  qword_5006BC8 = (__int64)&unk_5006BE0;
  qword_5006BD0 = 1;
  dword_5006BD8 = 0;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_5006BB0;
  if ( (unsigned __int64)(unsigned int)qword_5006BB0 + 1 > HIDWORD(qword_5006BB0) )
  {
    v31 = v9;
    sub_C8D5F0((char *)&unk_5006BB8 - 16, &unk_5006BB8, (unsigned int)qword_5006BB0 + 1LL, 8);
    v10 = (unsigned int)qword_5006BB0;
    v9 = v31;
  }
  *(_QWORD *)(qword_5006BA8 + 8 * v10) = v9;
  qword_5006BF0 = (__int64)&unk_49D9748;
  qword_5006B60 = (__int64)&unk_49DC090;
  qword_5006C00 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5006BB0) = qword_5006BB0 + 1;
  qword_5006C20 = (__int64)nullsub_23;
  qword_5006BE8 = 0;
  qword_5006C18 = (__int64)sub_984030;
  qword_5006BF8 = 0;
  sub_C53080(&qword_5006B60, "switch-range-to-icmp", 20);
  LOBYTE(qword_5006BE8) = 0;
  qword_5006B90 = 67;
  LOBYTE(dword_5006B6C) = dword_5006B6C & 0x9F | 0x20;
  LOWORD(qword_5006BF8) = 256;
  qword_5006B88 = (__int64)"Convert switches into an integer range comparison (default = false)";
  sub_C53130(&qword_5006B60);
  __cxa_atexit(sub_984900, &qword_5006B60, &qword_4A427C0);
  qword_5006A80 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_5006A8C &= 0x8000u;
  word_5006A90 = 0;
  qword_5006AD0 = 0x100000000LL;
  qword_5006AC8 = (__int64)&unk_5006AD8;
  qword_5006A98 = 0;
  qword_5006AA0 = 0;
  dword_5006A88 = v11;
  qword_5006AA8 = 0;
  qword_5006AB0 = 0;
  qword_5006AB8 = 0;
  qword_5006AC0 = 0;
  qword_5006AE0 = 0;
  qword_5006AE8 = (__int64)&unk_5006B00;
  qword_5006AF0 = 1;
  dword_5006AF8 = 0;
  byte_5006AFC = 1;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_5006AD0;
  if ( (unsigned __int64)(unsigned int)qword_5006AD0 + 1 > HIDWORD(qword_5006AD0) )
  {
    v32 = v12;
    sub_C8D5F0((char *)&unk_5006AD8 - 16, &unk_5006AD8, (unsigned int)qword_5006AD0 + 1LL, 8);
    v13 = (unsigned int)qword_5006AD0;
    v12 = v32;
  }
  *(_QWORD *)(qword_5006AC8 + 8 * v13) = v12;
  qword_5006B10 = (__int64)&unk_49D9748;
  qword_5006A80 = (__int64)&unk_49DC090;
  qword_5006B20 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5006AD0) = qword_5006AD0 + 1;
  qword_5006B40 = (__int64)nullsub_23;
  qword_5006B08 = 0;
  qword_5006B38 = (__int64)sub_984030;
  qword_5006B18 = 0;
  sub_C53080(&qword_5006A80, "switch-to-lookup", 16);
  LOBYTE(qword_5006B08) = 0;
  qword_5006AB0 = 51;
  LOBYTE(dword_5006A8C) = dword_5006A8C & 0x9F | 0x20;
  LOWORD(qword_5006B18) = 256;
  qword_5006AA8 = (__int64)"Convert switches to lookup tables (default = false)";
  sub_C53130(&qword_5006A80);
  __cxa_atexit(sub_984900, &qword_5006A80, &qword_4A427C0);
  qword_50069A0 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50069AC &= 0x8000u;
  word_50069B0 = 0;
  qword_50069F0 = 0x100000000LL;
  qword_50069E8 = (__int64)&unk_50069F8;
  qword_50069B8 = 0;
  qword_50069C0 = 0;
  dword_50069A8 = v14;
  qword_50069C8 = 0;
  qword_50069D0 = 0;
  qword_50069D8 = 0;
  qword_50069E0 = 0;
  qword_5006A00 = 0;
  qword_5006A08 = (__int64)&unk_5006A20;
  qword_5006A10 = 1;
  dword_5006A18 = 0;
  byte_5006A1C = 1;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_50069F0;
  if ( (unsigned __int64)(unsigned int)qword_50069F0 + 1 > HIDWORD(qword_50069F0) )
  {
    v33 = v15;
    sub_C8D5F0((char *)&unk_50069F8 - 16, &unk_50069F8, (unsigned int)qword_50069F0 + 1LL, 8);
    v16 = (unsigned int)qword_50069F0;
    v15 = v33;
  }
  *(_QWORD *)(qword_50069E8 + 8 * v16) = v15;
  qword_5006A30 = (__int64)&unk_49D9748;
  qword_50069A0 = (__int64)&unk_49DC090;
  qword_5006A40 = (__int64)&unk_49DC1D0;
  LODWORD(qword_50069F0) = qword_50069F0 + 1;
  qword_5006A60 = (__int64)nullsub_23;
  qword_5006A28 = 0;
  qword_5006A58 = (__int64)sub_984030;
  qword_5006A38 = 0;
  sub_C53080(&qword_50069A0, "forward-switch-cond", 19);
  LOWORD(qword_5006A38) = 256;
  LOBYTE(qword_5006A28) = 0;
  qword_50069D0 = 53;
  LOBYTE(dword_50069AC) = dword_50069AC & 0x9F | 0x20;
  qword_50069C8 = (__int64)"Forward switch condition to phi ops (default = false)";
  sub_C53130(&qword_50069A0);
  __cxa_atexit(sub_984900, &qword_50069A0, &qword_4A427C0);
  qword_50068C0 = (__int64)&unk_49DC150;
  v17 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5006910 = 0x100000000LL;
  dword_50068CC &= 0x8000u;
  qword_5006908 = (__int64)&unk_5006918;
  word_50068D0 = 0;
  qword_50068D8 = 0;
  dword_50068C8 = v17;
  qword_50068E0 = 0;
  qword_50068E8 = 0;
  qword_50068F0 = 0;
  qword_50068F8 = 0;
  qword_5006900 = 0;
  qword_5006920 = 0;
  qword_5006928 = (__int64)&unk_5006940;
  qword_5006930 = 1;
  dword_5006938 = 0;
  byte_500693C = 1;
  v18 = sub_C57470();
  v19 = (unsigned int)qword_5006910;
  if ( (unsigned __int64)(unsigned int)qword_5006910 + 1 > HIDWORD(qword_5006910) )
  {
    v34 = v18;
    sub_C8D5F0((char *)&unk_5006918 - 16, &unk_5006918, (unsigned int)qword_5006910 + 1LL, 8);
    v19 = (unsigned int)qword_5006910;
    v18 = v34;
  }
  *(_QWORD *)(qword_5006908 + 8 * v19) = v18;
  qword_5006950 = (__int64)&unk_49D9748;
  qword_50068C0 = (__int64)&unk_49DC090;
  qword_5006960 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5006910) = qword_5006910 + 1;
  qword_5006980 = (__int64)nullsub_23;
  qword_5006948 = 0;
  qword_5006978 = (__int64)sub_984030;
  qword_5006958 = 0;
  sub_C53080(&qword_50068C0, "hoist-common-insts", 18);
  LOWORD(qword_5006958) = 256;
  LOBYTE(qword_5006948) = 0;
  qword_50068F0 = 43;
  LOBYTE(dword_50068CC) = dword_50068CC & 0x9F | 0x20;
  qword_50068E8 = (__int64)"hoist common instructions (default = false)";
  sub_C53130(&qword_50068C0);
  __cxa_atexit(sub_984900, &qword_50068C0, &qword_4A427C0);
  qword_50067E0 = (__int64)&unk_49DC150;
  v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5006830 = 0x100000000LL;
  dword_50067EC &= 0x8000u;
  qword_5006828 = (__int64)&unk_5006838;
  word_50067F0 = 0;
  qword_50067F8 = 0;
  dword_50067E8 = v20;
  qword_5006800 = 0;
  qword_5006808 = 0;
  qword_5006810 = 0;
  qword_5006818 = 0;
  qword_5006820 = 0;
  qword_5006840 = 0;
  qword_5006848 = (__int64)&unk_5006860;
  qword_5006850 = 1;
  dword_5006858 = 0;
  byte_500685C = 1;
  v21 = sub_C57470();
  v22 = (unsigned int)qword_5006830;
  if ( (unsigned __int64)(unsigned int)qword_5006830 + 1 > HIDWORD(qword_5006830) )
  {
    v35 = v21;
    sub_C8D5F0((char *)&unk_5006838 - 16, &unk_5006838, (unsigned int)qword_5006830 + 1LL, 8);
    v22 = (unsigned int)qword_5006830;
    v21 = v35;
  }
  *(_QWORD *)(qword_5006828 + 8 * v22) = v21;
  qword_5006870 = (__int64)&unk_49D9748;
  qword_50067E0 = (__int64)&unk_49DC090;
  qword_5006880 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5006830) = qword_5006830 + 1;
  qword_50068A0 = (__int64)nullsub_23;
  qword_5006868 = 0;
  qword_5006898 = (__int64)sub_984030;
  qword_5006878 = 0;
  sub_C53080(&qword_50067E0, "hoist-loads-stores-with-cond-faulting", 37);
  LOWORD(qword_5006878) = 256;
  LOBYTE(qword_5006868) = 0;
  qword_5006810 = 80;
  LOBYTE(dword_50067EC) = dword_50067EC & 0x9F | 0x20;
  qword_5006808 = (__int64)"Hoist loads/stores if the target supports conditional faulting (default = false)";
  sub_C53130(&qword_50067E0);
  __cxa_atexit(sub_984900, &qword_50067E0, &qword_4A427C0);
  qword_5006700 = (__int64)&unk_49DC150;
  v23 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5006750 = 0x100000000LL;
  dword_500670C &= 0x8000u;
  qword_5006748 = (__int64)&unk_5006758;
  word_5006710 = 0;
  qword_5006718 = 0;
  dword_5006708 = v23;
  qword_5006720 = 0;
  qword_5006728 = 0;
  qword_5006730 = 0;
  qword_5006738 = 0;
  qword_5006740 = 0;
  qword_5006760 = 0;
  qword_5006768 = (__int64)&unk_5006780;
  qword_5006770 = 1;
  dword_5006778 = 0;
  byte_500677C = 1;
  v24 = sub_C57470();
  v25 = (unsigned int)qword_5006750;
  if ( (unsigned __int64)(unsigned int)qword_5006750 + 1 > HIDWORD(qword_5006750) )
  {
    v36 = v24;
    sub_C8D5F0((char *)&unk_5006758 - 16, &unk_5006758, (unsigned int)qword_5006750 + 1LL, 8);
    v25 = (unsigned int)qword_5006750;
    v24 = v36;
  }
  *(_QWORD *)(qword_5006748 + 8 * v25) = v24;
  qword_5006790 = (__int64)&unk_49D9748;
  qword_5006700 = (__int64)&unk_49DC090;
  qword_50067A0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5006750) = qword_5006750 + 1;
  qword_50067C0 = (__int64)nullsub_23;
  qword_5006788 = 0;
  qword_50067B8 = (__int64)sub_984030;
  qword_5006798 = 0;
  sub_C53080(&qword_5006700, "sink-common-insts", 17);
  LOWORD(qword_5006798) = 256;
  LOBYTE(qword_5006788) = 0;
  qword_5006730 = 42;
  LOBYTE(dword_500670C) = dword_500670C & 0x9F | 0x20;
  qword_5006728 = (__int64)"Sink common instructions (default = false)";
  sub_C53130(&qword_5006700);
  __cxa_atexit(sub_984900, &qword_5006700, &qword_4A427C0);
  qword_5006620 = (__int64)&unk_49DC150;
  v26 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5006670 = 0x100000000LL;
  dword_500662C &= 0x8000u;
  word_5006630 = 0;
  qword_5006668 = (__int64)&unk_5006678;
  qword_5006638 = 0;
  dword_5006628 = v26;
  qword_5006640 = 0;
  qword_5006648 = 0;
  qword_5006650 = 0;
  qword_5006658 = 0;
  qword_5006660 = 0;
  qword_5006680 = 0;
  qword_5006688 = (__int64)&unk_50066A0;
  qword_5006690 = 1;
  dword_5006698 = 0;
  byte_500669C = 1;
  v27 = sub_C57470();
  v28 = (unsigned int)qword_5006670;
  v29 = (unsigned int)qword_5006670 + 1LL;
  if ( v29 > HIDWORD(qword_5006670) )
  {
    sub_C8D5F0((char *)&unk_5006678 - 16, &unk_5006678, v29, 8);
    v28 = (unsigned int)qword_5006670;
  }
  *(_QWORD *)(qword_5006668 + 8 * v28) = v27;
  qword_50066B0 = (__int64)&unk_49D9748;
  qword_5006620 = (__int64)&unk_49DC090;
  qword_50066C0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5006670) = qword_5006670 + 1;
  qword_50066E0 = (__int64)nullsub_23;
  qword_50066A8 = 0;
  qword_50066D8 = (__int64)sub_984030;
  qword_50066B8 = 0;
  sub_C53080(&qword_5006620, "speculate-unpredictables", 24);
  LOBYTE(qword_50066A8) = 0;
  qword_5006650 = 50;
  LOBYTE(dword_500662C) = dword_500662C & 0x9F | 0x20;
  LOWORD(qword_50066B8) = 256;
  qword_5006648 = (__int64)"Speculate unpredictable branches (default = false)";
  sub_C53130(&qword_5006620);
  return __cxa_atexit(sub_984900, &qword_5006620, &qword_4A427C0);
}

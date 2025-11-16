// Function: ctor_525_0
// Address: 0x563730
//
int ctor_525_0()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
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
  __int64 v27; // rax
  __int64 v28; // rdx
  int v29; // edx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v33; // [rsp+8h] [rbp-38h]
  __int64 v34; // [rsp+8h] [rbp-38h]
  __int64 v35; // [rsp+8h] [rbp-38h]
  __int64 v36; // [rsp+8h] [rbp-38h]
  __int64 v37; // [rsp+8h] [rbp-38h]
  __int64 v38; // [rsp+8h] [rbp-38h]
  __int64 v39; // [rsp+8h] [rbp-38h]
  __int64 v40; // [rsp+8h] [rbp-38h]

  qword_5011A80 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5011AFC = 1;
  qword_5011AD0 = 0x100000000LL;
  dword_5011A8C &= 0x8000u;
  qword_5011A98 = 0;
  qword_5011AA0 = 0;
  qword_5011AA8 = 0;
  dword_5011A88 = v0;
  word_5011A90 = 0;
  qword_5011AB0 = 0;
  qword_5011AB8 = 0;
  qword_5011AC0 = 0;
  qword_5011AC8 = (__int64)&unk_5011AD8;
  qword_5011AE0 = 0;
  qword_5011AE8 = (__int64)&unk_5011B00;
  qword_5011AF0 = 1;
  dword_5011AF8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5011AD0;
  v3 = (unsigned int)qword_5011AD0 + 1LL;
  if ( v3 > HIDWORD(qword_5011AD0) )
  {
    sub_C8D5F0((char *)&unk_5011AD8 - 16, &unk_5011AD8, v3, 8);
    v2 = (unsigned int)qword_5011AD0;
  }
  *(_QWORD *)(qword_5011AC8 + 8 * v2) = v1;
  qword_5011B10 = (__int64)&unk_49D9728;
  LODWORD(qword_5011AD0) = qword_5011AD0 + 1;
  qword_5011B40 = (__int64)nullsub_24;
  qword_5011A80 = (__int64)&unk_49DBF10;
  qword_5011B08 = 0;
  qword_5011B18 = 0;
  qword_5011B20 = (__int64)&unk_49DC290;
  qword_5011B38 = (__int64)sub_984050;
  sub_C53080(&qword_5011A80, "dump-branch-dist", 16);
  LODWORD(qword_5011B08) = 0;
  BYTE4(qword_5011B18) = 1;
  LODWORD(qword_5011B18) = 0;
  qword_5011AB0 = 41;
  LOBYTE(dword_5011A8C) = dword_5011A8C & 0x9F | 0x20;
  qword_5011AA8 = (__int64)"Dump information from Branch Distribution";
  sub_C53130(&qword_5011A80);
  __cxa_atexit(sub_984970, &qword_5011A80, &qword_4A427C0);
  qword_50119A0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50119AC &= 0x8000u;
  word_50119B0 = 0;
  qword_50119F0 = 0x100000000LL;
  qword_50119B8 = 0;
  qword_50119C0 = 0;
  qword_50119C8 = 0;
  dword_50119A8 = v4;
  qword_50119D0 = 0;
  qword_50119D8 = 0;
  qword_50119E0 = 0;
  qword_50119E8 = (__int64)&unk_50119F8;
  qword_5011A00 = 0;
  qword_5011A08 = (__int64)&unk_5011A20;
  qword_5011A10 = 1;
  dword_5011A18 = 0;
  byte_5011A1C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50119F0;
  if ( (unsigned __int64)(unsigned int)qword_50119F0 + 1 > HIDWORD(qword_50119F0) )
  {
    v33 = v5;
    sub_C8D5F0((char *)&unk_50119F8 - 16, &unk_50119F8, (unsigned int)qword_50119F0 + 1LL, 8);
    v6 = (unsigned int)qword_50119F0;
    v5 = v33;
  }
  *(_QWORD *)(qword_50119E8 + 8 * v6) = v5;
  qword_5011A30 = (__int64)&unk_49D9728;
  LODWORD(qword_50119F0) = qword_50119F0 + 1;
  qword_5011A60 = (__int64)nullsub_24;
  qword_50119A0 = (__int64)&unk_49DBF10;
  qword_5011A28 = 0;
  qword_5011A38 = 0;
  qword_5011A40 = (__int64)&unk_49DC290;
  qword_5011A58 = (__int64)sub_984050;
  sub_C53080(&qword_50119A0, "ignore-call-safety", 18);
  LODWORD(qword_5011A28) = 1;
  BYTE4(qword_5011A38) = 1;
  LODWORD(qword_5011A38) = 1;
  qword_50119D0 = 42;
  LOBYTE(dword_50119AC) = dword_50119AC & 0x9F | 0x20;
  qword_50119C8 = (__int64)"Ignore calls safety in branch Distribution";
  sub_C53130(&qword_50119A0);
  __cxa_atexit(sub_984970, &qword_50119A0, &qword_4A427C0);
  qword_50118C0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50118CC &= 0x8000u;
  word_50118D0 = 0;
  qword_5011910 = 0x100000000LL;
  qword_50118D8 = 0;
  qword_50118E0 = 0;
  qword_50118E8 = 0;
  dword_50118C8 = v7;
  qword_50118F0 = 0;
  qword_50118F8 = 0;
  qword_5011900 = 0;
  qword_5011908 = (__int64)&unk_5011918;
  qword_5011920 = 0;
  qword_5011928 = (__int64)&unk_5011940;
  qword_5011930 = 1;
  dword_5011938 = 0;
  byte_501193C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_5011910;
  v10 = (unsigned int)qword_5011910 + 1LL;
  if ( v10 > HIDWORD(qword_5011910) )
  {
    sub_C8D5F0((char *)&unk_5011918 - 16, &unk_5011918, v10, 8);
    v9 = (unsigned int)qword_5011910;
  }
  *(_QWORD *)(qword_5011908 + 8 * v9) = v8;
  qword_5011950 = (__int64)&unk_49D9748;
  LODWORD(qword_5011910) = qword_5011910 + 1;
  qword_5011980 = (__int64)nullsub_23;
  qword_50118C0 = (__int64)&unk_49DC090;
  qword_5011960 = (__int64)&unk_49DC1D0;
  qword_5011948 = 0;
  qword_5011978 = (__int64)sub_984030;
  qword_5011958 = 0;
  sub_C53080(&qword_50118C0, "ignore-variance-cond", 20);
  LOBYTE(qword_5011948) = 0;
  LOWORD(qword_5011958) = 256;
  qword_50118F0 = 48;
  LOBYTE(dword_50118CC) = dword_50118CC & 0x9F | 0x20;
  qword_50118E8 = (__int64)"Ignore variance condition in branch Distribution";
  sub_C53130(&qword_50118C0);
  __cxa_atexit(sub_984900, &qword_50118C0, &qword_4A427C0);
  qword_50117E0 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50117EC &= 0x8000u;
  word_50117F0 = 0;
  qword_5011830 = 0x100000000LL;
  qword_5011828 = (__int64)&unk_5011838;
  qword_50117F8 = 0;
  qword_5011800 = 0;
  dword_50117E8 = v11;
  qword_5011808 = 0;
  qword_5011810 = 0;
  qword_5011818 = 0;
  qword_5011820 = 0;
  qword_5011840 = 0;
  qword_5011848 = (__int64)&unk_5011860;
  qword_5011850 = 1;
  dword_5011858 = 0;
  byte_501185C = 1;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_5011830;
  if ( (unsigned __int64)(unsigned int)qword_5011830 + 1 > HIDWORD(qword_5011830) )
  {
    v34 = v12;
    sub_C8D5F0((char *)&unk_5011838 - 16, &unk_5011838, (unsigned int)qword_5011830 + 1LL, 8);
    v13 = (unsigned int)qword_5011830;
    v12 = v34;
  }
  *(_QWORD *)(qword_5011828 + 8 * v13) = v12;
  qword_5011870 = (__int64)&unk_49D9748;
  LODWORD(qword_5011830) = qword_5011830 + 1;
  qword_50118A0 = (__int64)nullsub_23;
  qword_50117E0 = (__int64)&unk_49DC090;
  qword_5011880 = (__int64)&unk_49DC1D0;
  qword_5011868 = 0;
  qword_5011898 = (__int64)sub_984030;
  qword_5011878 = 0;
  sub_C53080(&qword_50117E0, "ignore-address-space-check", 26);
  LOWORD(qword_5011878) = 256;
  LOBYTE(qword_5011868) = 0;
  qword_5011810 = 50;
  LOBYTE(dword_50117EC) = dword_50117EC & 0x9F | 0x20;
  qword_5011808 = (__int64)"Ignore address-space checks in branch Distribution";
  sub_C53130(&qword_50117E0);
  __cxa_atexit(sub_984900, &qword_50117E0, &qword_4A427C0);
  qword_5011700 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_501170C &= 0x8000u;
  word_5011710 = 0;
  qword_5011750 = 0x100000000LL;
  qword_5011748 = (__int64)&unk_5011758;
  qword_5011718 = 0;
  qword_5011720 = 0;
  dword_5011708 = v14;
  qword_5011728 = 0;
  qword_5011730 = 0;
  qword_5011738 = 0;
  qword_5011740 = 0;
  qword_5011760 = 0;
  qword_5011768 = (__int64)&unk_5011780;
  qword_5011770 = 1;
  dword_5011778 = 0;
  byte_501177C = 1;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_5011750;
  if ( (unsigned __int64)(unsigned int)qword_5011750 + 1 > HIDWORD(qword_5011750) )
  {
    v35 = v15;
    sub_C8D5F0((char *)&unk_5011758 - 16, &unk_5011758, (unsigned int)qword_5011750 + 1LL, 8);
    v16 = (unsigned int)qword_5011750;
    v15 = v35;
  }
  *(_QWORD *)(qword_5011748 + 8 * v16) = v15;
  qword_5011790 = (__int64)&unk_49D9748;
  LODWORD(qword_5011750) = qword_5011750 + 1;
  qword_50117C0 = (__int64)nullsub_23;
  qword_5011700 = (__int64)&unk_49DC090;
  qword_50117A0 = (__int64)&unk_49DC1D0;
  qword_5011788 = 0;
  qword_50117B8 = (__int64)sub_984030;
  qword_5011798 = 0;
  sub_C53080(&qword_5011700, "ignore-phi-overhead", 19);
  LOWORD(qword_5011798) = 256;
  LOBYTE(qword_5011788) = 0;
  qword_5011730 = 31;
  LOBYTE(dword_501170C) = dword_501170C & 0x9F | 0x20;
  qword_5011728 = (__int64)"Ignore the overhead due to phis";
  sub_C53130(&qword_5011700);
  __cxa_atexit(sub_984900, &qword_5011700, &qword_4A427C0);
  qword_5011620 = (__int64)&unk_49DC150;
  v17 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5011670 = 0x100000000LL;
  dword_501162C &= 0x8000u;
  qword_5011668 = (__int64)&unk_5011678;
  word_5011630 = 0;
  qword_5011638 = 0;
  dword_5011628 = v17;
  qword_5011640 = 0;
  qword_5011648 = 0;
  qword_5011650 = 0;
  qword_5011658 = 0;
  qword_5011660 = 0;
  qword_5011680 = 0;
  qword_5011688 = (__int64)&unk_50116A0;
  qword_5011690 = 1;
  dword_5011698 = 0;
  byte_501169C = 1;
  v18 = sub_C57470();
  v19 = (unsigned int)qword_5011670;
  if ( (unsigned __int64)(unsigned int)qword_5011670 + 1 > HIDWORD(qword_5011670) )
  {
    v36 = v18;
    sub_C8D5F0((char *)&unk_5011678 - 16, &unk_5011678, (unsigned int)qword_5011670 + 1LL, 8);
    v19 = (unsigned int)qword_5011670;
    v18 = v36;
  }
  *(_QWORD *)(qword_5011668 + 8 * v19) = v18;
  qword_50116B0 = (__int64)&unk_49D9748;
  LODWORD(qword_5011670) = qword_5011670 + 1;
  qword_50116E0 = (__int64)nullsub_23;
  qword_5011620 = (__int64)&unk_49DC090;
  qword_50116C0 = (__int64)&unk_49DC1D0;
  qword_50116A8 = 0;
  qword_50116D8 = (__int64)sub_984030;
  qword_50116B8 = 0;
  sub_C53080(&qword_5011620, "disable-complex-branch-dist", 27);
  LOBYTE(qword_50116A8) = 0;
  LOWORD(qword_50116B8) = 256;
  qword_5011650 = 40;
  LOBYTE(dword_501162C) = dword_501162C & 0x9F | 0x20;
  qword_5011648 = (__int64)"Disable more complex branch Distribution";
  sub_C53130(&qword_5011620);
  __cxa_atexit(sub_984900, &qword_5011620, &qword_4A427C0);
  qword_5011520 = (__int64)&unk_49DC150;
  v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  word_5011530 = 0;
  qword_5011568 = (__int64)&unk_5011578;
  qword_5011538 = 0;
  dword_501152C = dword_501152C & 0x8000 | 1;
  qword_5011570 = 0x100000000LL;
  dword_5011528 = v20;
  qword_5011540 = 0;
  qword_5011548 = 0;
  qword_5011550 = 0;
  qword_5011558 = 0;
  qword_5011560 = 0;
  qword_5011580 = 0;
  qword_5011588 = (__int64)&unk_50115A0;
  qword_5011590 = 1;
  dword_5011598 = 0;
  byte_501159C = 1;
  v21 = sub_C57470();
  v22 = (unsigned int)qword_5011570;
  if ( (unsigned __int64)(unsigned int)qword_5011570 + 1 > HIDWORD(qword_5011570) )
  {
    v37 = v21;
    sub_C8D5F0((char *)&unk_5011578 - 16, &unk_5011578, (unsigned int)qword_5011570 + 1LL, 8);
    v22 = (unsigned int)qword_5011570;
    v21 = v37;
  }
  *(_QWORD *)(qword_5011568 + 8 * v22) = v21;
  LODWORD(qword_5011570) = qword_5011570 + 1;
  qword_50115A8 = 0;
  qword_5011520 = (__int64)&unk_49DAD08;
  qword_50115B0 = 0;
  qword_50115B8 = 0;
  qword_50115F8 = (__int64)&unk_49DC350;
  qword_50115C0 = 0;
  qword_5011618 = (__int64)nullsub_81;
  qword_50115C8 = 0;
  qword_5011610 = (__int64)sub_BB8600;
  qword_50115D0 = 0;
  byte_50115D8 = 0;
  qword_50115E0 = 0;
  qword_50115E8 = 0;
  qword_50115F0 = 0;
  sub_C53080(&qword_5011520, "no-branch-dist", 14);
  BYTE1(dword_501152C) |= 2u;
  qword_5011548 = (__int64)"Do not do Branch Distribution on some functions";
  qword_5011558 = (__int64)"function1,function2,,...";
  qword_5011550 = 47;
  qword_5011560 = 24;
  sub_C53130(&qword_5011520);
  __cxa_atexit(sub_BB89D0, &qword_5011520, &qword_4A427C0);
  qword_5011440 = (__int64)&unk_49DC150;
  v23 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_50114BC = 1;
  word_5011450 = 0;
  qword_5011490 = 0x100000000LL;
  dword_501144C &= 0x8000u;
  qword_5011488 = (__int64)&unk_5011498;
  qword_5011458 = 0;
  dword_5011448 = v23;
  qword_5011460 = 0;
  qword_5011468 = 0;
  qword_5011470 = 0;
  qword_5011478 = 0;
  qword_5011480 = 0;
  qword_50114A0 = 0;
  qword_50114A8 = (__int64)&unk_50114C0;
  qword_50114B0 = 1;
  dword_50114B8 = 0;
  v24 = sub_C57470();
  v25 = (unsigned int)qword_5011490;
  if ( (unsigned __int64)(unsigned int)qword_5011490 + 1 > HIDWORD(qword_5011490) )
  {
    v38 = v24;
    sub_C8D5F0((char *)&unk_5011498 - 16, &unk_5011498, (unsigned int)qword_5011490 + 1LL, 8);
    v25 = (unsigned int)qword_5011490;
    v24 = v38;
  }
  *(_QWORD *)(qword_5011488 + 8 * v25) = v24;
  LODWORD(qword_5011490) = qword_5011490 + 1;
  qword_50114C8 = 0;
  qword_50114D0 = (__int64)&unk_49DA090;
  qword_50114D8 = 0;
  qword_5011440 = (__int64)&unk_49DBF90;
  qword_50114E0 = (__int64)&unk_49DC230;
  qword_5011500 = (__int64)nullsub_58;
  qword_50114F8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_5011440, "branch-dist-func-limit", 22);
  LODWORD(qword_50114C8) = -1;
  qword_5011468 = (__int64)"Control number of functions to apply";
  BYTE4(qword_50114D8) = 1;
  LODWORD(qword_50114D8) = -1;
  qword_5011470 = 36;
  sub_C53130(&qword_5011440);
  __cxa_atexit(sub_B2B680, &qword_5011440, &qword_4A427C0);
  qword_5011360 = (__int64)&unk_49DC150;
  v26 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50113B0 = 0x100000000LL;
  dword_501136C &= 0x8000u;
  word_5011370 = 0;
  qword_50113A8 = (__int64)&unk_50113B8;
  qword_5011378 = 0;
  dword_5011368 = v26;
  qword_5011380 = 0;
  qword_5011388 = 0;
  qword_5011390 = 0;
  qword_5011398 = 0;
  qword_50113A0 = 0;
  qword_50113C0 = 0;
  qword_50113C8 = (__int64)&unk_50113E0;
  qword_50113D0 = 1;
  dword_50113D8 = 0;
  byte_50113DC = 1;
  v27 = sub_C57470();
  v28 = (unsigned int)qword_50113B0;
  if ( (unsigned __int64)(unsigned int)qword_50113B0 + 1 > HIDWORD(qword_50113B0) )
  {
    v39 = v27;
    sub_C8D5F0((char *)&unk_50113B8 - 16, &unk_50113B8, (unsigned int)qword_50113B0 + 1LL, 8);
    v28 = (unsigned int)qword_50113B0;
    v27 = v39;
  }
  *(_QWORD *)(qword_50113A8 + 8 * v28) = v27;
  LODWORD(qword_50113B0) = qword_50113B0 + 1;
  qword_50113E8 = 0;
  qword_50113F0 = (__int64)&unk_49DA090;
  qword_50113F8 = 0;
  qword_5011360 = (__int64)&unk_49DBF90;
  qword_5011400 = (__int64)&unk_49DC230;
  qword_5011420 = (__int64)nullsub_58;
  qword_5011418 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_5011360, "branch-dist-block-limit", 23);
  LODWORD(qword_50113E8) = -1;
  qword_5011388 = (__int64)"Control number of blocks to apply";
  BYTE4(qword_50113F8) = 1;
  LODWORD(qword_50113F8) = -1;
  qword_5011390 = 33;
  sub_C53130(&qword_5011360);
  __cxa_atexit(sub_B2B680, &qword_5011360, &qword_4A427C0);
  qword_5011280 = (__int64)&unk_49DC150;
  v29 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_50112FC = 1;
  qword_50112D0 = 0x100000000LL;
  dword_501128C &= 0x8000u;
  qword_50112C8 = (__int64)&unk_50112D8;
  qword_5011298 = 0;
  qword_50112A0 = 0;
  dword_5011288 = v29;
  word_5011290 = 0;
  qword_50112A8 = 0;
  qword_50112B0 = 0;
  qword_50112B8 = 0;
  qword_50112C0 = 0;
  qword_50112E0 = 0;
  qword_50112E8 = (__int64)&unk_5011300;
  qword_50112F0 = 1;
  dword_50112F8 = 0;
  v30 = sub_C57470();
  v31 = (unsigned int)qword_50112D0;
  if ( (unsigned __int64)(unsigned int)qword_50112D0 + 1 > HIDWORD(qword_50112D0) )
  {
    v40 = v30;
    sub_C8D5F0((char *)&unk_50112D8 - 16, &unk_50112D8, (unsigned int)qword_50112D0 + 1LL, 8);
    v31 = (unsigned int)qword_50112D0;
    v30 = v40;
  }
  *(_QWORD *)(qword_50112C8 + 8 * v31) = v30;
  qword_5011310 = (__int64)&unk_49D9748;
  qword_5011340 = (__int64)nullsub_23;
  LODWORD(qword_50112D0) = qword_50112D0 + 1;
  qword_5011280 = (__int64)&unk_49DC090;
  qword_5011320 = (__int64)&unk_49DC1D0;
  qword_5011308 = 0;
  qword_5011338 = (__int64)sub_984030;
  qword_5011318 = 0;
  sub_C53080(&qword_5011280, "branch-dist-norm", 16);
  LOBYTE(qword_5011308) = 0;
  LOWORD(qword_5011318) = 256;
  qword_50112A8 = (__int64)"Control normalization for branch dist";
  qword_50112B0 = 37;
  sub_C53130(&qword_5011280);
  return __cxa_atexit(sub_984900, &qword_5011280, &qword_4A427C0);
}

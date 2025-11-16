// Function: ctor_472_0
// Address: 0x54b6b0
//
int ctor_472_0()
{
  int v0; // edx
  __int64 v1; // r15
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // r15
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  int v12; // edx
  __int64 v13; // r15
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  int v16; // edx
  __int64 v17; // r15
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  int v20; // edx
  __int64 v21; // r15
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  int v24; // edx
  __int64 v25; // rbx
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  int v28; // edx
  __int64 v29; // rbx
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  int v33; // [rsp+40h] [rbp-50h] BYREF
  int v34; // [rsp+44h] [rbp-4Ch] BYREF
  int *v35; // [rsp+48h] [rbp-48h] BYREF
  const char *v36; // [rsp+50h] [rbp-40h] BYREF
  __int64 v37; // [rsp+58h] [rbp-38h]

  sub_D95050(qword_5003700, 0, 0);
  qword_5003700[17] = 0;
  qword_5003700[19] = 0;
  qword_5003700[18] = &unk_49D9748;
  qword_5003700[0] = &unk_49DC090;
  qword_5003700[20] = &unk_49DC1D0;
  qword_5003700[24] = nullsub_23;
  qword_5003700[23] = sub_984030;
  sub_C53080(qword_5003700, "forget-scev-loop-unroll", 23);
  LOBYTE(qword_5003700[17]) = 0;
  LOWORD(qword_5003700[19]) = 256;
  qword_5003700[6] = 143;
  BYTE4(qword_5003700[1]) = BYTE4(qword_5003700[1]) & 0x9F | 0x20;
  qword_5003700[5] = "Forget everything in SCEV when doing LoopUnroll, instead of just the current top-most loop. This is"
                     " sometimes preferred to reduce compile time.";
  sub_C53130(qword_5003700);
  __cxa_atexit(sub_984900, qword_5003700, &qword_4A427C0);
  v36 = "The cost threshold for loop unrolling";
  v37 = 37;
  LODWORD(v35) = 1;
  sub_28818A0(&unk_5003620, "unroll-threshold", &v35, &v36);
  __cxa_atexit(sub_984970, &unk_5003620, &qword_4A427C0);
  v36 = "The cost threshold for loop unrolling when optimizing for size";
  v35 = &v33;
  v37 = 62;
  v34 = 1;
  v33 = 0;
  sub_2881A90(&unk_5003540, "unroll-optsize-threshold", &v35, &v34, &v36);
  __cxa_atexit(sub_984970, &unk_5003540, &qword_4A427C0);
  qword_5003460 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_500346C &= 0x8000u;
  word_5003470 = 0;
  qword_50034B0 = 0x100000000LL;
  qword_5003478 = 0;
  qword_5003480 = 0;
  qword_5003488 = 0;
  dword_5003468 = v0;
  qword_5003490 = 0;
  qword_5003498 = 0;
  qword_50034A0 = 0;
  qword_50034A8 = (__int64)&unk_50034B8;
  qword_50034C0 = 0;
  qword_50034C8 = (__int64)&unk_50034E0;
  qword_50034D0 = 1;
  dword_50034D8 = 0;
  byte_50034DC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_50034B0;
  v3 = (unsigned int)qword_50034B0 + 1LL;
  if ( v3 > HIDWORD(qword_50034B0) )
  {
    sub_C8D5F0((char *)&unk_50034B8 - 16, &unk_50034B8, v3, 8);
    v2 = (unsigned int)qword_50034B0;
  }
  *(_QWORD *)(qword_50034A8 + 8 * v2) = v1;
  LODWORD(qword_50034B0) = qword_50034B0 + 1;
  qword_50034E8 = 0;
  qword_50034F0 = (__int64)&unk_49D9728;
  qword_5003460 = (__int64)&unk_49DBF10;
  qword_5003500 = (__int64)&unk_49DC290;
  qword_5003520 = (__int64)nullsub_24;
  qword_50034F8 = 0;
  qword_5003518 = (__int64)sub_984050;
  sub_C53080(&qword_5003460, "unroll-partial-threshold", 24);
  qword_5003490 = 45;
  LOBYTE(dword_500346C) = dword_500346C & 0x9F | 0x20;
  qword_5003488 = (__int64)"The cost threshold for partial loop unrolling";
  sub_C53130(&qword_5003460);
  __cxa_atexit(sub_984970, &qword_5003460, &qword_4A427C0);
  sub_D95050(&qword_5003380, 0, 0);
  qword_5003420 = (__int64)&unk_49DC290;
  qword_5003440 = (__int64)nullsub_24;
  qword_5003410 = (__int64)&unk_49D9728;
  qword_5003380 = (__int64)&unk_49DBF10;
  qword_5003438 = (__int64)sub_984050;
  qword_5003408 = 0;
  qword_5003418 = 0;
  sub_C53080(&qword_5003380, "unroll-max-percent-threshold-boost", 34);
  LODWORD(qword_5003408) = 400;
  BYTE4(qword_5003418) = 1;
  LODWORD(qword_5003418) = 400;
  qword_50033B0 = 359;
  byte_500338C = byte_500338C & 0x9F | 0x20;
  qword_50033A8 = (__int64)"The maximum 'boost' (represented as a percentage >= 100) applied to the threshold when aggres"
                           "sively unrolling a loop due to the dynamic cost savings. If completely unrolling a loop will "
                           "reduce the total runtime from X to Y, we boost the loop unroll threshold to DefaultThreshold*"
                           "std::min(MaxPercentThresholdBoost, X/Y). This limit avoids excessive code bloat.";
  sub_C53130(&qword_5003380);
  __cxa_atexit(sub_984970, &qword_5003380, &qword_4A427C0);
  qword_50032A0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50032AC &= 0x8000u;
  word_50032B0 = 0;
  qword_50032F0 = 0x100000000LL;
  qword_50032E8 = (__int64)&unk_50032F8;
  qword_50032B8 = 0;
  qword_50032C0 = 0;
  dword_50032A8 = v4;
  qword_50032C8 = 0;
  qword_50032D0 = 0;
  qword_50032D8 = 0;
  qword_50032E0 = 0;
  qword_5003300 = 0;
  qword_5003308 = (__int64)&unk_5003320;
  qword_5003310 = 1;
  dword_5003318 = 0;
  byte_500331C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50032F0;
  v7 = (unsigned int)qword_50032F0 + 1LL;
  if ( v7 > HIDWORD(qword_50032F0) )
  {
    sub_C8D5F0((char *)&unk_50032F8 - 16, &unk_50032F8, v7, 8);
    v6 = (unsigned int)qword_50032F0;
  }
  *(_QWORD *)(qword_50032E8 + 8 * v6) = v5;
  LODWORD(qword_50032F0) = qword_50032F0 + 1;
  qword_5003328 = 0;
  qword_5003330 = (__int64)&unk_49D9728;
  qword_50032A0 = (__int64)&unk_49DBF10;
  qword_5003340 = (__int64)&unk_49DC290;
  qword_5003360 = (__int64)nullsub_24;
  qword_5003338 = 0;
  qword_5003358 = (__int64)sub_984050;
  sub_C53080(&qword_50032A0, "unroll-max-iteration-count-to-analyze", 37);
  LODWORD(qword_5003328) = 10;
  BYTE4(qword_5003338) = 1;
  LODWORD(qword_5003338) = 10;
  qword_50032D0 = 114;
  LOBYTE(dword_50032AC) = dword_50032AC & 0x9F | 0x20;
  qword_50032C8 = (__int64)"Don't allow loop unrolling to simulate more than this number of iterations when checking full"
                           " unroll profitability";
  sub_C53130(&qword_50032A0);
  __cxa_atexit(sub_984970, &qword_50032A0, &qword_4A427C0);
  sub_D95050(&qword_50031C0, 0, 0);
  qword_5003260 = (__int64)&unk_49DC290;
  qword_5003280 = (__int64)nullsub_24;
  qword_5003250 = (__int64)&unk_49D9728;
  qword_50031C0 = (__int64)&unk_49DBF10;
  qword_5003278 = (__int64)sub_984050;
  qword_5003248 = 0;
  qword_5003258 = 0;
  sub_C53080(&qword_50031C0, "unroll-count", 12);
  qword_50031F0 = 105;
  byte_50031CC = byte_50031CC & 0x9F | 0x20;
  qword_50031E8 = (__int64)"Use this unroll count for all loops including those with unroll_count pragma values, for testing purposes";
  sub_C53130(&qword_50031C0);
  __cxa_atexit(sub_984970, &qword_50031C0, &qword_4A427C0);
  v36 = "Set the max unroll count for partial and runtime unrolling, fortesting purposes";
  v37 = 79;
  LODWORD(v35) = 1;
  sub_28818A0(&unk_50030E0, "unroll-max-count", &v35, &v36);
  __cxa_atexit(sub_984970, &unk_50030E0, &qword_4A427C0);
  sub_D95050(&qword_5003000, 0, 0);
  qword_50030A0 = (__int64)&unk_49DC290;
  qword_50030C0 = (__int64)nullsub_24;
  qword_5003090 = (__int64)&unk_49D9728;
  qword_5003000 = (__int64)&unk_49DBF10;
  qword_50030B8 = (__int64)sub_984050;
  qword_5003088 = 0;
  qword_5003098 = 0;
  sub_C53080(&qword_5003000, "unroll-full-max-count", 21);
  qword_5003030 = 65;
  byte_500300C = byte_500300C & 0x9F | 0x20;
  qword_5003028 = (__int64)"Set the max unroll count for full unrolling, for testing purposes";
  sub_C53130(&qword_5003000);
  __cxa_atexit(sub_984970, &qword_5003000, &qword_4A427C0);
  sub_D95050(&qword_5002F20, 0, 0);
  qword_5002FA8 = 0;
  qword_5002FB8 = 0;
  qword_5002FB0 = (__int64)&unk_49D9748;
  qword_5002F20 = (__int64)&unk_49DC090;
  qword_5002FC0 = (__int64)&unk_49DC1D0;
  qword_5002FE0 = (__int64)nullsub_23;
  qword_5002FD8 = (__int64)sub_984030;
  sub_C53080(&qword_5002F20, "unroll-allow-partial", 20);
  qword_5002F50 = 83;
  byte_5002F2C = byte_5002F2C & 0x9F | 0x20;
  qword_5002F48 = (__int64)"Allows loops to be partially unrolled until -unroll-threshold loop size is reached.";
  sub_C53130(&qword_5002F20);
  __cxa_atexit(sub_984900, &qword_5002F20, &qword_4A427C0);
  sub_D95050(&qword_5002E40, 0, 0);
  qword_5002EE0 = (__int64)&unk_49DC1D0;
  qword_5002F00 = (__int64)nullsub_23;
  qword_5002ED0 = (__int64)&unk_49D9748;
  qword_5002E40 = (__int64)&unk_49DC090;
  qword_5002EF8 = (__int64)sub_984030;
  qword_5002EC8 = 0;
  qword_5002ED8 = 0;
  sub_C53080(&qword_5002E40, "unroll-allow-remainder", 22);
  qword_5002E70 = 78;
  byte_5002E4C = byte_5002E4C & 0x9F | 0x20;
  qword_5002E68 = (__int64)"Allow generation of a loop remainder (extra iterations) when unrolling a loop.";
  sub_C53130(&qword_5002E40);
  __cxa_atexit(sub_984900, &qword_5002E40, &qword_4A427C0);
  qword_5002D60 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_5002D6C &= 0x8000u;
  word_5002D70 = 0;
  qword_5002DB0 = 0x100000000LL;
  qword_5002DA8 = (__int64)&unk_5002DB8;
  qword_5002D78 = 0;
  qword_5002D80 = 0;
  dword_5002D68 = v8;
  qword_5002D88 = 0;
  qword_5002D90 = 0;
  qword_5002D98 = 0;
  qword_5002DA0 = 0;
  qword_5002DC0 = 0;
  qword_5002DC8 = (__int64)&unk_5002DE0;
  qword_5002DD0 = 1;
  dword_5002DD8 = 0;
  byte_5002DDC = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_5002DB0;
  v11 = (unsigned int)qword_5002DB0 + 1LL;
  if ( v11 > HIDWORD(qword_5002DB0) )
  {
    sub_C8D5F0((char *)&unk_5002DB8 - 16, &unk_5002DB8, v11, 8);
    v10 = (unsigned int)qword_5002DB0;
  }
  *(_QWORD *)(qword_5002DA8 + 8 * v10) = v9;
  LODWORD(qword_5002DB0) = qword_5002DB0 + 1;
  qword_5002DE8 = 0;
  qword_5002DF0 = (__int64)&unk_49D9748;
  qword_5002D60 = (__int64)&unk_49DC090;
  qword_5002E00 = (__int64)&unk_49DC1D0;
  qword_5002E20 = (__int64)nullsub_23;
  qword_5002DF8 = 0;
  qword_5002E18 = (__int64)sub_984030;
  sub_C53080(&qword_5002D60, "unroll-runtime", 14);
  qword_5002D90 = 38;
  LOBYTE(dword_5002D6C) = dword_5002D6C & 0x9F | 0x20;
  qword_5002D88 = (__int64)"Unroll loops with run-time trip counts";
  sub_C53130(&qword_5002D60);
  __cxa_atexit(sub_984900, &qword_5002D60, &qword_4A427C0);
  sub_D95050(&qword_5002C80, 0, 0);
  qword_5002D08 = 0;
  qword_5002D18 = 0;
  qword_5002D10 = (__int64)&unk_49D9728;
  qword_5002C80 = (__int64)&unk_49DBF10;
  qword_5002D40 = (__int64)nullsub_24;
  qword_5002D20 = (__int64)&unk_49DC290;
  qword_5002D38 = (__int64)sub_984050;
  sub_C53080(&qword_5002C80, "unroll-max-upperbound", 21);
  LODWORD(qword_5002D08) = 8;
  BYTE4(qword_5002D18) = 1;
  LODWORD(qword_5002D18) = 8;
  qword_5002CB0 = 65;
  byte_5002C8C = byte_5002C8C & 0x9F | 0x20;
  qword_5002CA8 = (__int64)"The max of trip count upper bound that is considered in unrolling";
  sub_C53130(&qword_5002C80);
  __cxa_atexit(sub_984970, &qword_5002C80, &qword_4A427C0);
  sub_D95050(&qword_5002BA0, 0, 0);
  qword_5002C60 = (__int64)nullsub_24;
  qword_5002C30 = (__int64)&unk_49D9728;
  qword_5002BA0 = (__int64)&unk_49DBF10;
  qword_5002C40 = (__int64)&unk_49DC290;
  qword_5002C58 = (__int64)sub_984050;
  qword_5002C28 = 0;
  qword_5002C38 = 0;
  sub_C53080(&qword_5002BA0, "pragma-unroll-threshold", 23);
  LODWORD(qword_5002C28) = 0x8000;
  BYTE4(qword_5002C38) = 1;
  LODWORD(qword_5002C38) = 0x8000;
  qword_5002BD0 = 74;
  byte_5002BAC = byte_5002BAC & 0x9F | 0x20;
  qword_5002BC8 = (__int64)"Unrolled size limit for loops with an unroll(full) or unroll_count pragma.";
  sub_C53130(&qword_5002BA0);
  __cxa_atexit(sub_984970, &qword_5002BA0, &qword_4A427C0);
  v37 = 137;
  v36 = "If the runtime tripcount for the loop is lower than the threshold, the loop is considered as flat and will be le"
        "ss aggressively unrolled.";
  v34 = 1;
  v35 = &v33;
  v33 = 5;
  sub_2881CB0(&unk_5002AC0, "flat-loop-tripcount-threshold", &v35, &v34, &v36);
  __cxa_atexit(sub_984970, &unk_5002AC0, &qword_4A427C0);
  sub_D95050(&qword_50029E0, 0, 0);
  qword_5002A80 = (__int64)&unk_49DC1D0;
  qword_5002AA0 = (__int64)nullsub_23;
  qword_5002A70 = (__int64)&unk_49D9748;
  qword_50029E0 = (__int64)&unk_49DC090;
  qword_5002A98 = (__int64)sub_984030;
  qword_5002A68 = 0;
  qword_5002A78 = 0;
  sub_C53080(&qword_50029E0, "unroll-count-extern-indirect-call-as-inline", 43);
  LOWORD(qword_5002A78) = 256;
  LOBYTE(qword_5002A68) = 0;
  qword_5002A10 = 135;
  byte_50029EC = byte_50029EC & 0x9F | 0x20;
  qword_5002A08 = (__int64)"During unroll loop analysis, consider calls to extern functions or indirect calls as potentia"
                           "lly inlinable (e.g. during LTO generation)";
  sub_C53130(&qword_50029E0);
  __cxa_atexit(sub_984900, &qword_50029E0, &qword_4A427C0);
  qword_5002900 = (__int64)&unk_49DC150;
  v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5002950 = 0x100000000LL;
  dword_500290C &= 0x8000u;
  word_5002910 = 0;
  qword_5002948 = (__int64)&unk_5002958;
  qword_5002918 = 0;
  dword_5002908 = v12;
  qword_5002920 = 0;
  qword_5002928 = 0;
  qword_5002930 = 0;
  qword_5002938 = 0;
  qword_5002940 = 0;
  qword_5002960 = 0;
  qword_5002968 = (__int64)&unk_5002980;
  qword_5002970 = 1;
  dword_5002978 = 0;
  byte_500297C = 1;
  v13 = sub_C57470();
  v14 = (unsigned int)qword_5002950;
  v15 = (unsigned int)qword_5002950 + 1LL;
  if ( v15 > HIDWORD(qword_5002950) )
  {
    sub_C8D5F0((char *)&unk_5002958 - 16, &unk_5002958, v15, 8);
    v14 = (unsigned int)qword_5002950;
  }
  *(_QWORD *)(qword_5002948 + 8 * v14) = v13;
  LODWORD(qword_5002950) = qword_5002950 + 1;
  qword_5002988 = 0;
  qword_5002990 = (__int64)&unk_49D9748;
  qword_5002900 = (__int64)&unk_49DC090;
  qword_50029A0 = (__int64)&unk_49DC1D0;
  qword_50029C0 = (__int64)nullsub_23;
  qword_5002998 = 0;
  qword_50029B8 = (__int64)sub_984030;
  sub_C53080(&qword_5002900, "unroll-remainder", 16);
  qword_5002930 = 40;
  LOBYTE(dword_500290C) = dword_500290C & 0x9F | 0x20;
  qword_5002928 = (__int64)"Allow the loop remainder to be unrolled.";
  sub_C53130(&qword_5002900);
  __cxa_atexit(sub_984900, &qword_5002900, &qword_4A427C0);
  sub_D95050(&qword_5002820, 0, 0);
  qword_50028C0 = (__int64)&unk_49DC1D0;
  qword_50028E0 = (__int64)nullsub_23;
  qword_50028B0 = (__int64)&unk_49D9748;
  qword_5002820 = (__int64)&unk_49DC090;
  qword_50028D8 = (__int64)sub_984030;
  qword_50028A8 = 0;
  qword_50028B8 = 0;
  sub_C53080(&qword_5002820, "unroll-remainder-auto", 21);
  LOWORD(qword_50028B8) = 257;
  LOBYTE(qword_50028A8) = 1;
  qword_5002850 = 42;
  byte_500282C = byte_500282C & 0x9F | 0x20;
  qword_5002848 = (__int64)"Auto unroll the remainder for inner loops.";
  sub_C53130(&qword_5002820);
  __cxa_atexit(sub_984900, &qword_5002820, &qword_4A427C0);
  sub_D95050(&qword_5002740, 0, 0);
  qword_50027E0 = (__int64)&unk_49DC1D0;
  qword_5002800 = (__int64)nullsub_23;
  qword_50027D0 = (__int64)&unk_49D9748;
  qword_5002740 = (__int64)&unk_49DC090;
  qword_50027F8 = (__int64)sub_984030;
  qword_50027C8 = 0;
  qword_50027D8 = 0;
  sub_C53080(&qword_5002740, "unroll-revisit-child-loops", 26);
  qword_5002770 = 154;
  byte_500274C = byte_500274C & 0x9F | 0x20;
  qword_5002768 = (__int64)"Enqueue and re-visit child loops in the loop PM after unrolling. This shouldn't typically be "
                           "needed as child loops (or their clones) were already visited.";
  sub_C53130(&qword_5002740);
  __cxa_atexit(sub_984900, &qword_5002740, &qword_4A427C0);
  sub_D95050(&qword_5002660, 0, 0);
  qword_50026E8 = 0;
  qword_50026F8 = 0;
  qword_50026F0 = (__int64)&unk_49D9728;
  qword_5002660 = (__int64)&unk_49DBF10;
  qword_5002700 = (__int64)&unk_49DC290;
  qword_5002720 = (__int64)nullsub_24;
  qword_5002718 = (__int64)sub_984050;
  sub_C53080(&qword_5002660, "unroll-threshold-aggressive", 27);
  LODWORD(qword_50026E8) = 405;
  BYTE4(qword_50026F8) = 1;
  LODWORD(qword_50026F8) = 405;
  qword_5002690 = 77;
  byte_500266C = byte_500266C & 0x9F | 0x20;
  qword_5002688 = (__int64)"Threshold (max size of unrolled loop) to use in aggressive (O3) optimizations";
  sub_C53130(&qword_5002660);
  __cxa_atexit(sub_984970, &qword_5002660, &qword_4A427C0);
  v37 = 79;
  v36 = "Default threshold (max size of unrolled loop), used in all but O3 optimizations";
  v34 = 1;
  v35 = &v33;
  v33 = 150;
  sub_2881A90(&unk_5002580, "unroll-threshold-default", &v35, &v34, &v36);
  __cxa_atexit(sub_984970, &unk_5002580, &qword_4A427C0);
  qword_50024A0 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50024F0 = 0x100000000LL;
  dword_50024AC &= 0x8000u;
  qword_50024E8 = (__int64)&unk_50024F8;
  word_50024B0 = 0;
  qword_50024B8 = 0;
  dword_50024A8 = v16;
  qword_50024C0 = 0;
  qword_50024C8 = 0;
  qword_50024D0 = 0;
  qword_50024D8 = 0;
  qword_50024E0 = 0;
  qword_5002500 = 0;
  qword_5002508 = (__int64)&unk_5002520;
  qword_5002510 = 1;
  dword_5002518 = 0;
  byte_500251C = 1;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_50024F0;
  v19 = (unsigned int)qword_50024F0 + 1LL;
  if ( v19 > HIDWORD(qword_50024F0) )
  {
    sub_C8D5F0((char *)&unk_50024F8 - 16, &unk_50024F8, v19, 8);
    v18 = (unsigned int)qword_50024F0;
  }
  *(_QWORD *)(qword_50024E8 + 8 * v18) = v17;
  LODWORD(qword_50024F0) = qword_50024F0 + 1;
  qword_5002528 = 0;
  qword_5002530 = (__int64)&unk_49D9728;
  qword_5002538 = 0;
  qword_50024A0 = (__int64)&unk_49DBF10;
  qword_5002540 = (__int64)&unk_49DC290;
  qword_5002560 = (__int64)nullsub_24;
  qword_5002558 = (__int64)sub_984050;
  sub_C53080(&qword_50024A0, "pragma-unroll-full-max-iterations", 33);
  LODWORD(qword_5002528) = 1000000;
  BYTE4(qword_5002538) = 1;
  LODWORD(qword_5002538) = 1000000;
  qword_50024D0 = 62;
  LOBYTE(dword_50024AC) = dword_50024AC & 0x9F | 0x20;
  qword_50024C8 = (__int64)"Maximum allowed iterations to unroll under pragma unroll full.";
  sub_C53130(&qword_50024A0);
  __cxa_atexit(sub_984970, &qword_50024A0, &qword_4A427C0);
  qword_50023C0 = (__int64)&unk_49DC150;
  v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5002410 = 0x100000000LL;
  dword_50023CC &= 0x8000u;
  qword_5002408 = (__int64)&unk_5002418;
  word_50023D0 = 0;
  qword_50023D8 = 0;
  dword_50023C8 = v20;
  qword_50023E0 = 0;
  qword_50023E8 = 0;
  qword_50023F0 = 0;
  qword_50023F8 = 0;
  qword_5002400 = 0;
  qword_5002420 = 0;
  qword_5002428 = (__int64)&unk_5002440;
  qword_5002430 = 1;
  dword_5002438 = 0;
  byte_500243C = 1;
  v21 = sub_C57470();
  v22 = (unsigned int)qword_5002410;
  v23 = (unsigned int)qword_5002410 + 1LL;
  if ( v23 > HIDWORD(qword_5002410) )
  {
    sub_C8D5F0((char *)&unk_5002418 - 16, &unk_5002418, v23, 8);
    v22 = (unsigned int)qword_5002410;
  }
  *(_QWORD *)(qword_5002408 + 8 * v22) = v21;
  LODWORD(qword_5002410) = qword_5002410 + 1;
  qword_5002448 = 0;
  qword_5002450 = (__int64)&unk_49D9728;
  qword_5002458 = 0;
  qword_50023C0 = (__int64)&unk_49DBF10;
  qword_5002460 = (__int64)&unk_49DC290;
  qword_5002480 = (__int64)nullsub_24;
  qword_5002478 = (__int64)sub_984050;
  sub_C53080(&qword_50023C0, "max-pragma-upperbound-unroll", 28);
  LODWORD(qword_5002448) = 64;
  BYTE4(qword_5002458) = 1;
  LODWORD(qword_5002458) = 64;
  qword_50023F0 = 77;
  LOBYTE(dword_50023CC) = dword_50023CC & 0x9F | 0x20;
  qword_50023E8 = (__int64)"The max of trip count upper bound that is considered in unrolling with pragma";
  sub_C53130(&qword_50023C0);
  __cxa_atexit(sub_984970, &qword_50023C0, &qword_4A427C0);
  v37 = 49;
  v36 = "The cut-off point for automatic runtime unrolling";
  v34 = 1;
  v33 = 95;
  v35 = &v33;
  sub_2881A90(&unk_50022E0, "runtime-unroll-threshold", &v35, &v34, &v36);
  __cxa_atexit(sub_984970, &unk_50022E0, &qword_4A427C0);
  v37 = 69;
  v36 = "The maximum iteration count below which runtime unrolling is disabled";
  v34 = 1;
  v33 = 20;
  v35 = &v33;
  sub_2881CB0(&unk_5002200, "runtime-unroll-iter-threshold", &v35, &v34, &v36);
  __cxa_atexit(sub_984970, &unk_5002200, &qword_4A427C0);
  qword_5002120 = (__int64)&unk_49DC150;
  v24 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5002170 = 0x100000000LL;
  dword_500212C &= 0x8000u;
  word_5002130 = 0;
  qword_5002138 = 0;
  qword_5002140 = 0;
  dword_5002128 = v24;
  qword_5002148 = 0;
  qword_5002150 = 0;
  qword_5002158 = 0;
  qword_5002160 = 0;
  qword_5002168 = (__int64)&unk_5002178;
  qword_5002180 = 0;
  qword_5002188 = (__int64)&unk_50021A0;
  qword_5002190 = 1;
  dword_5002198 = 0;
  byte_500219C = 1;
  v25 = sub_C57470();
  v26 = (unsigned int)qword_5002170;
  v27 = (unsigned int)qword_5002170 + 1LL;
  if ( v27 > HIDWORD(qword_5002170) )
  {
    sub_C8D5F0((char *)&unk_5002178 - 16, &unk_5002178, v27, 8);
    v26 = (unsigned int)qword_5002170;
  }
  *(_QWORD *)(qword_5002168 + 8 * v26) = v25;
  LODWORD(qword_5002170) = qword_5002170 + 1;
  qword_50021A8 = 0;
  qword_50021B0 = (__int64)&unk_49D9748;
  qword_50021B8 = 0;
  qword_5002120 = (__int64)&unk_49DC090;
  qword_50021C0 = (__int64)&unk_49DC1D0;
  qword_50021E0 = (__int64)nullsub_23;
  qword_50021D8 = (__int64)sub_984030;
  sub_C53080(&qword_5002120, "runtime-unroll-check-profit", 27);
  LOWORD(qword_50021B8) = 257;
  LOBYTE(qword_50021A8) = 1;
  qword_5002150 = 59;
  LOBYTE(dword_500212C) = dword_500212C & 0x9F | 0x20;
  qword_5002148 = (__int64)"Check if the input loop is profitable for runtime unrolling";
  sub_C53130(&qword_5002120);
  __cxa_atexit(sub_984900, &qword_5002120, &qword_4A427C0);
  sub_D95050(&qword_5002040, 0, 0);
  qword_50020D0 = (__int64)&unk_49D9748;
  qword_5002100 = (__int64)nullsub_23;
  qword_5002040 = (__int64)&unk_49DC090;
  qword_50020E0 = (__int64)&unk_49DC1D0;
  qword_50020F8 = (__int64)sub_984030;
  qword_50020C8 = 0;
  qword_50020D8 = 0;
  sub_C53080(&qword_5002040, "aggressive-runtime-unrolling", 28);
  LOWORD(qword_50020D8) = 257;
  LOBYTE(qword_50020C8) = 1;
  qword_5002070 = 144;
  byte_500204C = byte_500204C & 0x9F | 0x20;
  qword_5002068 = (__int64)"During unroll loop analysis, consider unrolling loops according to OCG unrolling heuristics w"
                           "hich is more aggressive with loops containing loads";
  sub_C53130(&qword_5002040);
  __cxa_atexit(sub_984900, &qword_5002040, &qword_4A427C0);
  sub_D95050(&qword_5001F60, 0, 0);
  qword_5001FE8 = 0;
  qword_5001FF8 = 0;
  qword_5001FF0 = (__int64)&unk_49D9728;
  qword_5001F60 = (__int64)&unk_49DBF10;
  qword_5002000 = (__int64)&unk_49DC290;
  qword_5002020 = (__int64)nullsub_24;
  qword_5002018 = (__int64)sub_984050;
  sub_C53080(&qword_5001F60, "aggressive-runtime-unrolling-fixed-factor", 41);
  LODWORD(qword_5001FE8) = 0;
  BYTE4(qword_5001FF8) = 1;
  LODWORD(qword_5001FF8) = 0;
  qword_5001F90 = 296;
  byte_5001F6C = byte_5001F6C & 0x9F | 0x20;
  qword_5001F88 = (__int64)"Option to force the aggressive runtime unroller to always give out the same unroll factor, ra"
                           "ther than a range of possible values. This is to support the use cases of other teams that do"
                           "n't rely on any NVVM unrolling functionality, and want to emulate the original OCG functional"
                           "ity more closely.";
  sub_C53130(&qword_5001F60);
  __cxa_atexit(sub_984970, &qword_5001F60, &qword_4A427C0);
  sub_D95050(&qword_5001E80, 0, 0);
  qword_5001F10 = (__int64)&unk_49D9728;
  qword_5001F40 = (__int64)nullsub_24;
  qword_5001E80 = (__int64)&unk_49DBF10;
  qword_5001F20 = (__int64)&unk_49DC290;
  qword_5001F38 = (__int64)sub_984050;
  qword_5001F08 = 0;
  qword_5001F18 = 0;
  sub_C53080(&qword_5001E80, "aggressive-runtime-unrolling-max-factor", 39);
  LODWORD(qword_5001F08) = 16;
  BYTE4(qword_5001F18) = 1;
  LODWORD(qword_5001F18) = 16;
  qword_5001EB0 = 81;
  byte_5001E8C = byte_5001E8C & 0x9F | 0x20;
  qword_5001EA8 = (__int64)"The maximum possible unroll factor that the aggressive runtime unroller can emit.";
  sub_C53130(&qword_5001E80);
  __cxa_atexit(sub_984970, &qword_5001E80, &qword_4A427C0);
  sub_D95050(&qword_5001DA0, 0, 0);
  qword_5001E30 = (__int64)&unk_49D9728;
  qword_5001E60 = (__int64)nullsub_24;
  qword_5001DA0 = (__int64)&unk_49DBF10;
  qword_5001E40 = (__int64)&unk_49DC290;
  qword_5001E58 = (__int64)sub_984050;
  qword_5001E28 = 0;
  qword_5001E38 = 0;
  sub_C53080(&qword_5001DA0, "aggressive-runtime-unrolling-max-filler-instructions-per-batch", 62);
  LODWORD(qword_5001E28) = 220;
  BYTE4(qword_5001E38) = 1;
  LODWORD(qword_5001E38) = 220;
  qword_5001DD0 = 106;
  byte_5001DAC = byte_5001DAC & 0x9F | 0x20;
  qword_5001DC8 = (__int64)"For aggressively unrolled runtime loops, the maximum amount of instructions that the unrolled loop can be.";
  sub_C53130(&qword_5001DA0);
  __cxa_atexit(sub_984970, &qword_5001DA0, &qword_4A427C0);
  qword_5001CC0 = (__int64)&unk_49DC150;
  v28 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5001D10 = 0x100000000LL;
  dword_5001CCC &= 0x8000u;
  word_5001CD0 = 0;
  qword_5001CD8 = 0;
  qword_5001CE0 = 0;
  dword_5001CC8 = v28;
  qword_5001CE8 = 0;
  qword_5001CF0 = 0;
  qword_5001CF8 = 0;
  qword_5001D00 = 0;
  qword_5001D08 = (__int64)&unk_5001D18;
  qword_5001D20 = 0;
  qword_5001D28 = (__int64)&unk_5001D40;
  qword_5001D30 = 1;
  dword_5001D38 = 0;
  byte_5001D3C = 1;
  v29 = sub_C57470();
  v30 = (unsigned int)qword_5001D10;
  v31 = (unsigned int)qword_5001D10 + 1LL;
  if ( v31 > HIDWORD(qword_5001D10) )
  {
    sub_C8D5F0((char *)&unk_5001D18 - 16, &unk_5001D18, v31, 8);
    v30 = (unsigned int)qword_5001D10;
  }
  *(_QWORD *)(qword_5001D08 + 8 * v30) = v29;
  LODWORD(qword_5001D10) = qword_5001D10 + 1;
  qword_5001D48 = 0;
  qword_5001D50 = (__int64)&unk_49D9748;
  qword_5001D58 = 0;
  qword_5001CC0 = (__int64)&unk_49DC090;
  qword_5001D60 = (__int64)&unk_49DC1D0;
  qword_5001D80 = (__int64)nullsub_23;
  qword_5001D78 = (__int64)sub_984030;
  sub_C53080(&qword_5001CC0, "waterfall-unrolling", 19);
  LOBYTE(qword_5001D48) = 1;
  LOWORD(qword_5001D58) = 257;
  qword_5001CF0 = 83;
  LOBYTE(dword_5001CCC) = dword_5001CCC & 0x9F | 0x20;
  qword_5001CE8 = (__int64)"For runtime unrolled loops that are profitable for waterfall unrolling, perform it.";
  sub_C53130(&qword_5001CC0);
  return __cxa_atexit(sub_984900, &qword_5001CC0, &qword_4A427C0);
}

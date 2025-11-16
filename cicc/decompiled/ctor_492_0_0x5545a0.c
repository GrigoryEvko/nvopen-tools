// Function: ctor_492_0
// Address: 0x5545a0
//
int ctor_492_0()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  int v3; // edx
  __int64 v4; // rax
  __int64 v5; // rdx
  int v6; // edx
  __int64 v7; // rbx
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // r14
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // r14
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  int v18; // edx
  __int64 v19; // r14
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  int v22; // edx
  __int64 v23; // r14
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  int v26; // edx
  __int64 v27; // r14
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  int v30; // edx
  __int64 v31; // r14
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  int v34; // edx
  __int64 v35; // rax
  __int64 v36; // rdx
  int v37; // edx
  __int64 v38; // rax
  __int64 v39; // rdx
  int v40; // edx
  __int64 v41; // rax
  __int64 v42; // rdx
  int v43; // edx
  __int64 v44; // rax
  __int64 v45; // rdx
  int v46; // edx
  __int64 v47; // rax
  __int64 v48; // rdx
  int v49; // edx
  __int64 v50; // rax
  __int64 v51; // rdx
  int v52; // edx
  __int64 v53; // rax
  __int64 v54; // rdx
  int v55; // edx
  __int64 v56; // r14
  __int64 v57; // rax
  unsigned __int64 v58; // rdx
  int v59; // edx
  __int64 v60; // r14
  __int64 v61; // rax
  unsigned __int64 v62; // rdx
  __int64 v64; // [rsp+8h] [rbp-38h]
  __int64 v65; // [rsp+8h] [rbp-38h]
  __int64 v66; // [rsp+8h] [rbp-38h]
  __int64 v67; // [rsp+8h] [rbp-38h]
  __int64 v68; // [rsp+8h] [rbp-38h]
  __int64 v69; // [rsp+8h] [rbp-38h]
  __int64 v70; // [rsp+8h] [rbp-38h]
  __int64 v71; // [rsp+8h] [rbp-38h]

  qword_5008A20 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_5008A2C = word_5008A2C & 0x8000;
  qword_5008A68[1] = 0x100000000LL;
  unk_5008A28 = v0;
  unk_5008A30 = 0;
  unk_5008A38 = 0;
  unk_5008A40 = 0;
  unk_5008A48 = 0;
  unk_5008A50 = 0;
  unk_5008A58 = 0;
  unk_5008A60 = 0;
  qword_5008A68[0] = &qword_5008A68[2];
  qword_5008A68[3] = 0;
  qword_5008A68[4] = &qword_5008A68[7];
  qword_5008A68[5] = 1;
  LODWORD(qword_5008A68[6]) = 0;
  BYTE4(qword_5008A68[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_5008A68[1]);
  if ( (unsigned __int64)LODWORD(qword_5008A68[1]) + 1 > HIDWORD(qword_5008A68[1]) )
  {
    sub_C8D5F0(qword_5008A68, &qword_5008A68[2], LODWORD(qword_5008A68[1]) + 1LL, 8);
    v2 = LODWORD(qword_5008A68[1]);
  }
  *(_QWORD *)(qword_5008A68[0] + 8 * v2) = v1;
  ++LODWORD(qword_5008A68[1]);
  qword_5008A68[8] = 0;
  qword_5008A68[9] = &unk_49D9748;
  qword_5008A68[15] = nullsub_23;
  qword_5008A68[10] = 0;
  qword_5008A20 = &unk_49DC090;
  qword_5008A68[11] = &unk_49DC1D0;
  qword_5008A68[14] = sub_984030;
  sub_C53080(&qword_5008A20, "enable-ext-tsp-block-placement", 30);
  LOWORD(qword_5008A68[10]) = 256;
  LOBYTE(qword_5008A68[8]) = 0;
  unk_5008A50 = 90;
  LOBYTE(word_5008A2C) = word_5008A2C & 0x9F | 0x20;
  unk_5008A48 = "Enable machine block placement based on the ext-tsp model, optimizing I-cache utilization.";
  sub_C53130(&qword_5008A20);
  __cxa_atexit(sub_984900, &qword_5008A20, &qword_4A427C0);
  qword_5008940 = &unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_500894C = word_500894C & 0x8000;
  qword_5008988[1] = 0x100000000LL;
  unk_5008948 = v3;
  unk_5008950 = 0;
  unk_5008958 = 0;
  unk_5008960 = 0;
  unk_5008968 = 0;
  unk_5008970 = 0;
  unk_5008978 = 0;
  unk_5008980 = 0;
  qword_5008988[0] = &qword_5008988[2];
  qword_5008988[3] = 0;
  qword_5008988[4] = &qword_5008988[7];
  qword_5008988[5] = 1;
  LODWORD(qword_5008988[6]) = 0;
  BYTE4(qword_5008988[6]) = 1;
  v4 = sub_C57470();
  v5 = LODWORD(qword_5008988[1]);
  if ( (unsigned __int64)LODWORD(qword_5008988[1]) + 1 > HIDWORD(qword_5008988[1]) )
  {
    v64 = v4;
    sub_C8D5F0(qword_5008988, &qword_5008988[2], LODWORD(qword_5008988[1]) + 1LL, 8);
    v5 = LODWORD(qword_5008988[1]);
    v4 = v64;
  }
  *(_QWORD *)(qword_5008988[0] + 8 * v5) = v4;
  ++LODWORD(qword_5008988[1]);
  qword_5008988[15] = nullsub_23;
  qword_5008988[9] = &unk_49D9748;
  qword_5008988[8] = 0;
  qword_5008988[10] = 0;
  qword_5008940 = &unk_49DC090;
  qword_5008988[11] = &unk_49DC1D0;
  qword_5008988[14] = sub_984030;
  sub_C53080(&qword_5008940, "ext-tsp-apply-without-profile", 29);
  unk_5008968 = "Whether to apply ext-tsp placement for instances w/o profile";
  LOWORD(qword_5008988[10]) = 257;
  unk_5008970 = 60;
  LOBYTE(qword_5008988[8]) = 1;
  LOBYTE(word_500894C) = word_500894C & 0x9F | 0x20;
  sub_C53130(&qword_5008940);
  __cxa_atexit(sub_984900, &qword_5008940, &qword_4A427C0);
  qword_5008860 = (__int64)&unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50088B0 = 0x100000000LL;
  word_5008870 = 0;
  dword_500886C &= 0x8000u;
  qword_5008878 = 0;
  qword_5008880 = 0;
  dword_5008868 = v6;
  qword_5008888 = 0;
  qword_5008890 = 0;
  qword_5008898 = 0;
  qword_50088A0 = 0;
  qword_50088A8 = (__int64)&unk_50088B8;
  qword_50088C0 = 0;
  qword_50088C8 = (__int64)&unk_50088E0;
  qword_50088D0 = 1;
  dword_50088D8 = 0;
  byte_50088DC = 1;
  v7 = sub_C57470();
  v8 = (unsigned int)qword_50088B0;
  v9 = (unsigned int)qword_50088B0 + 1LL;
  if ( v9 > HIDWORD(qword_50088B0) )
  {
    sub_C8D5F0((char *)&unk_50088B8 - 16, &unk_50088B8, v9, 8);
    v8 = (unsigned int)qword_50088B0;
  }
  *(_QWORD *)(qword_50088A8 + 8 * v8) = v7;
  qword_50088F0 = (__int64)&unk_49DE5F0;
  LODWORD(qword_50088B0) = qword_50088B0 + 1;
  qword_5008860 = (__int64)&unk_49DE610;
  qword_50088E8 = 0;
  byte_5008900 = 0;
  qword_5008908 = (__int64)&unk_49DC2F0;
  qword_50088F8 = 0;
  qword_5008928 = (__int64)nullsub_190;
  qword_5008920 = (__int64)sub_D83E80;
  sub_C53080(&qword_5008860, "ext-tsp-forward-weight-cond", 27);
  byte_5008900 = 1;
  qword_5008890 = 56;
  LOBYTE(dword_500886C) = dword_500886C & 0x9F | 0x40;
  qword_50088E8 = 0x3FB999999999999ALL;
  qword_50088F8 = 0x3FB999999999999ALL;
  qword_5008888 = (__int64)"The weight of conditional forward jumps for ExtTSP value";
  sub_C53130(&qword_5008860);
  __cxa_atexit(sub_D84280, &qword_5008860, &qword_4A427C0);
  qword_5008780 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_500878C &= 0x8000u;
  word_5008790 = 0;
  qword_50087D0 = 0x100000000LL;
  qword_50087C8 = (__int64)&unk_50087D8;
  qword_5008798 = 0;
  qword_50087A0 = 0;
  dword_5008788 = v10;
  qword_50087A8 = 0;
  qword_50087B0 = 0;
  qword_50087B8 = 0;
  qword_50087C0 = 0;
  qword_50087E0 = 0;
  qword_50087E8 = (__int64)&unk_5008800;
  qword_50087F0 = 1;
  dword_50087F8 = 0;
  byte_50087FC = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_50087D0;
  v13 = (unsigned int)qword_50087D0 + 1LL;
  if ( v13 > HIDWORD(qword_50087D0) )
  {
    sub_C8D5F0((char *)&unk_50087D8 - 16, &unk_50087D8, v13, 8);
    v12 = (unsigned int)qword_50087D0;
  }
  *(_QWORD *)(qword_50087C8 + 8 * v12) = v11;
  qword_5008810 = (__int64)&unk_49DE5F0;
  qword_5008780 = (__int64)&unk_49DE610;
  LODWORD(qword_50087D0) = qword_50087D0 + 1;
  byte_5008820 = 0;
  qword_5008828 = (__int64)&unk_49DC2F0;
  qword_5008808 = 0;
  qword_5008848 = (__int64)nullsub_190;
  qword_5008818 = 0;
  qword_5008840 = (__int64)sub_D83E80;
  sub_C53080(&qword_5008780, "ext-tsp-forward-weight-uncond", 29);
  byte_5008820 = 1;
  qword_50087B0 = 58;
  LOBYTE(dword_500878C) = dword_500878C & 0x9F | 0x40;
  qword_5008808 = 0x3FB999999999999ALL;
  qword_5008818 = 0x3FB999999999999ALL;
  qword_50087A8 = (__int64)"The weight of unconditional forward jumps for ExtTSP value";
  sub_C53130(&qword_5008780);
  __cxa_atexit(sub_D84280, &qword_5008780, &qword_4A427C0);
  qword_50086A0 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50086AC &= 0x8000u;
  word_50086B0 = 0;
  qword_50086F0 = 0x100000000LL;
  qword_50086E8 = (__int64)&unk_50086F8;
  qword_50086B8 = 0;
  qword_50086C0 = 0;
  dword_50086A8 = v14;
  qword_50086C8 = 0;
  qword_50086D0 = 0;
  qword_50086D8 = 0;
  qword_50086E0 = 0;
  qword_5008700 = 0;
  qword_5008708 = (__int64)&unk_5008720;
  qword_5008710 = 1;
  dword_5008718 = 0;
  byte_500871C = 1;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_50086F0;
  v17 = (unsigned int)qword_50086F0 + 1LL;
  if ( v17 > HIDWORD(qword_50086F0) )
  {
    sub_C8D5F0((char *)&unk_50086F8 - 16, &unk_50086F8, v17, 8);
    v16 = (unsigned int)qword_50086F0;
  }
  *(_QWORD *)(qword_50086E8 + 8 * v16) = v15;
  qword_5008730 = (__int64)&unk_49DE5F0;
  qword_50086A0 = (__int64)&unk_49DE610;
  LODWORD(qword_50086F0) = qword_50086F0 + 1;
  byte_5008740 = 0;
  qword_5008748 = (__int64)&unk_49DC2F0;
  qword_5008728 = 0;
  qword_5008768 = (__int64)nullsub_190;
  qword_5008738 = 0;
  qword_5008760 = (__int64)sub_D83E80;
  sub_C53080(&qword_50086A0, "ext-tsp-backward-weight-cond", 28);
  byte_5008740 = 1;
  qword_50086D0 = 57;
  LOBYTE(dword_50086AC) = dword_50086AC & 0x9F | 0x40;
  qword_5008728 = 0x3FB999999999999ALL;
  qword_5008738 = 0x3FB999999999999ALL;
  qword_50086C8 = (__int64)"The weight of conditional backward jumps for ExtTSP value";
  sub_C53130(&qword_50086A0);
  __cxa_atexit(sub_D84280, &qword_50086A0, &qword_4A427C0);
  qword_50085C0 = (__int64)&unk_49DC150;
  v18 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50085CC &= 0x8000u;
  word_50085D0 = 0;
  qword_5008610 = 0x100000000LL;
  qword_5008608 = (__int64)&unk_5008618;
  qword_50085D8 = 0;
  qword_50085E0 = 0;
  dword_50085C8 = v18;
  qword_50085E8 = 0;
  qword_50085F0 = 0;
  qword_50085F8 = 0;
  qword_5008600 = 0;
  qword_5008620 = 0;
  qword_5008628 = (__int64)&unk_5008640;
  qword_5008630 = 1;
  dword_5008638 = 0;
  byte_500863C = 1;
  v19 = sub_C57470();
  v20 = (unsigned int)qword_5008610;
  v21 = (unsigned int)qword_5008610 + 1LL;
  if ( v21 > HIDWORD(qword_5008610) )
  {
    sub_C8D5F0((char *)&unk_5008618 - 16, &unk_5008618, v21, 8);
    v20 = (unsigned int)qword_5008610;
  }
  *(_QWORD *)(qword_5008608 + 8 * v20) = v19;
  qword_5008650 = (__int64)&unk_49DE5F0;
  qword_50085C0 = (__int64)&unk_49DE610;
  LODWORD(qword_5008610) = qword_5008610 + 1;
  byte_5008660 = 0;
  qword_5008668 = (__int64)&unk_49DC2F0;
  qword_5008648 = 0;
  qword_5008688 = (__int64)nullsub_190;
  qword_5008658 = 0;
  qword_5008680 = (__int64)sub_D83E80;
  sub_C53080(&qword_50085C0, "ext-tsp-backward-weight-uncond", 30);
  byte_5008660 = 1;
  qword_50085F0 = 59;
  LOBYTE(dword_50085CC) = dword_50085CC & 0x9F | 0x40;
  qword_5008648 = 0x3FB999999999999ALL;
  qword_5008658 = 0x3FB999999999999ALL;
  qword_50085E8 = (__int64)"The weight of unconditional backward jumps for ExtTSP value";
  sub_C53130(&qword_50085C0);
  __cxa_atexit(sub_D84280, &qword_50085C0, &qword_4A427C0);
  qword_50084E0 = (__int64)&unk_49DC150;
  v22 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50084EC &= 0x8000u;
  word_50084F0 = 0;
  qword_5008530 = 0x100000000LL;
  qword_5008528 = (__int64)&unk_5008538;
  qword_50084F8 = 0;
  qword_5008500 = 0;
  dword_50084E8 = v22;
  qword_5008508 = 0;
  qword_5008510 = 0;
  qword_5008518 = 0;
  qword_5008520 = 0;
  qword_5008540 = 0;
  qword_5008548 = (__int64)&unk_5008560;
  qword_5008550 = 1;
  dword_5008558 = 0;
  byte_500855C = 1;
  v23 = sub_C57470();
  v24 = (unsigned int)qword_5008530;
  v25 = (unsigned int)qword_5008530 + 1LL;
  if ( v25 > HIDWORD(qword_5008530) )
  {
    sub_C8D5F0((char *)&unk_5008538 - 16, &unk_5008538, v25, 8);
    v24 = (unsigned int)qword_5008530;
  }
  *(_QWORD *)(qword_5008528 + 8 * v24) = v23;
  qword_5008570 = (__int64)&unk_49DE5F0;
  qword_50084E0 = (__int64)&unk_49DE610;
  LODWORD(qword_5008530) = qword_5008530 + 1;
  byte_5008580 = 0;
  qword_5008588 = (__int64)&unk_49DC2F0;
  qword_5008568 = 0;
  qword_50085A8 = (__int64)nullsub_190;
  qword_5008578 = 0;
  qword_50085A0 = (__int64)sub_D83E80;
  sub_C53080(&qword_50084E0, "ext-tsp-fallthrough-weight-cond", 31);
  byte_5008580 = 1;
  qword_5008568 = 0x3FF0000000000000LL;
  qword_5008578 = 0x3FF0000000000000LL;
  LOBYTE(dword_50084EC) = dword_50084EC & 0x9F | 0x40;
  qword_5008508 = (__int64)"The weight of conditional fallthrough jumps for ExtTSP value";
  qword_5008510 = 60;
  sub_C53130(&qword_50084E0);
  __cxa_atexit(sub_D84280, &qword_50084E0, &qword_4A427C0);
  qword_5008400 = (__int64)&unk_49DC150;
  v26 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_500840C &= 0x8000u;
  word_5008410 = 0;
  qword_5008450 = 0x100000000LL;
  qword_5008448 = (__int64)&unk_5008458;
  qword_5008418 = 0;
  qword_5008420 = 0;
  dword_5008408 = v26;
  qword_5008428 = 0;
  qword_5008430 = 0;
  qword_5008438 = 0;
  qword_5008440 = 0;
  qword_5008460 = 0;
  qword_5008468 = (__int64)&unk_5008480;
  qword_5008470 = 1;
  dword_5008478 = 0;
  byte_500847C = 1;
  v27 = sub_C57470();
  v28 = (unsigned int)qword_5008450;
  v29 = (unsigned int)qword_5008450 + 1LL;
  if ( v29 > HIDWORD(qword_5008450) )
  {
    sub_C8D5F0((char *)&unk_5008458 - 16, &unk_5008458, v29, 8);
    v28 = (unsigned int)qword_5008450;
  }
  *(_QWORD *)(qword_5008448 + 8 * v28) = v27;
  qword_5008490 = (__int64)&unk_49DE5F0;
  qword_5008400 = (__int64)&unk_49DE610;
  LODWORD(qword_5008450) = qword_5008450 + 1;
  byte_50084A0 = 0;
  qword_50084A8 = (__int64)&unk_49DC2F0;
  qword_5008488 = 0;
  qword_50084C8 = (__int64)nullsub_190;
  qword_5008498 = 0;
  qword_50084C0 = (__int64)sub_D83E80;
  sub_C53080(&qword_5008400, "ext-tsp-fallthrough-weight-uncond", 33);
  byte_50084A0 = 1;
  qword_5008488 = 0x3FF0CCCCCCCCCCCDLL;
  qword_5008498 = 0x3FF0CCCCCCCCCCCDLL;
  LOBYTE(dword_500840C) = dword_500840C & 0x9F | 0x40;
  qword_5008428 = (__int64)"The weight of unconditional fallthrough jumps for ExtTSP value";
  qword_5008430 = 62;
  sub_C53130(&qword_5008400);
  __cxa_atexit(sub_D84280, &qword_5008400, &qword_4A427C0);
  qword_5008320 = (__int64)&unk_49DC150;
  v30 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5008370 = 0x100000000LL;
  dword_500832C &= 0x8000u;
  word_5008330 = 0;
  qword_5008368 = (__int64)&unk_5008378;
  qword_5008338 = 0;
  dword_5008328 = v30;
  qword_5008340 = 0;
  qword_5008348 = 0;
  qword_5008350 = 0;
  qword_5008358 = 0;
  qword_5008360 = 0;
  qword_5008380 = 0;
  qword_5008388 = (__int64)&unk_50083A0;
  qword_5008390 = 1;
  dword_5008398 = 0;
  byte_500839C = 1;
  v31 = sub_C57470();
  v32 = (unsigned int)qword_5008370;
  v33 = (unsigned int)qword_5008370 + 1LL;
  if ( v33 > HIDWORD(qword_5008370) )
  {
    sub_C8D5F0((char *)&unk_5008378 - 16, &unk_5008378, v33, 8);
    v32 = (unsigned int)qword_5008370;
  }
  *(_QWORD *)(qword_5008368 + 8 * v32) = v31;
  LODWORD(qword_5008370) = qword_5008370 + 1;
  qword_50083A8 = 0;
  qword_50083B0 = (__int64)&unk_49D9728;
  qword_50083B8 = 0;
  qword_5008320 = (__int64)&unk_49DBF10;
  qword_50083C0 = (__int64)&unk_49DC290;
  qword_50083E0 = (__int64)nullsub_24;
  qword_50083D8 = (__int64)sub_984050;
  sub_C53080(&qword_5008320, "ext-tsp-forward-distance", 24);
  LODWORD(qword_50083A8) = 1024;
  BYTE4(qword_50083B8) = 1;
  LODWORD(qword_50083B8) = 1024;
  qword_5008350 = 60;
  LOBYTE(dword_500832C) = dword_500832C & 0x9F | 0x40;
  qword_5008348 = (__int64)"The maximum distance (in bytes) of a forward jump for ExtTSP";
  sub_C53130(&qword_5008320);
  __cxa_atexit(sub_984970, &qword_5008320, &qword_4A427C0);
  qword_5008240 = (__int64)&unk_49DC150;
  v34 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5008290 = 0x100000000LL;
  dword_500824C &= 0x8000u;
  qword_5008288 = (__int64)&unk_5008298;
  word_5008250 = 0;
  qword_5008258 = 0;
  dword_5008248 = v34;
  qword_5008260 = 0;
  qword_5008268 = 0;
  qword_5008270 = 0;
  qword_5008278 = 0;
  qword_5008280 = 0;
  qword_50082A0 = 0;
  qword_50082A8 = (__int64)&unk_50082C0;
  qword_50082B0 = 1;
  dword_50082B8 = 0;
  byte_50082BC = 1;
  v35 = sub_C57470();
  v36 = (unsigned int)qword_5008290;
  if ( (unsigned __int64)(unsigned int)qword_5008290 + 1 > HIDWORD(qword_5008290) )
  {
    v65 = v35;
    sub_C8D5F0((char *)&unk_5008298 - 16, &unk_5008298, (unsigned int)qword_5008290 + 1LL, 8);
    v36 = (unsigned int)qword_5008290;
    v35 = v65;
  }
  *(_QWORD *)(qword_5008288 + 8 * v36) = v35;
  LODWORD(qword_5008290) = qword_5008290 + 1;
  qword_50082C8 = 0;
  qword_50082D0 = (__int64)&unk_49D9728;
  qword_50082D8 = 0;
  qword_5008240 = (__int64)&unk_49DBF10;
  qword_50082E0 = (__int64)&unk_49DC290;
  qword_5008300 = (__int64)nullsub_24;
  qword_50082F8 = (__int64)sub_984050;
  sub_C53080(&qword_5008240, "ext-tsp-backward-distance", 25);
  LODWORD(qword_50082C8) = 640;
  BYTE4(qword_50082D8) = 1;
  LODWORD(qword_50082D8) = 640;
  qword_5008270 = 61;
  LOBYTE(dword_500824C) = dword_500824C & 0x9F | 0x40;
  qword_5008268 = (__int64)"The maximum distance (in bytes) of a backward jump for ExtTSP";
  sub_C53130(&qword_5008240);
  __cxa_atexit(sub_984970, &qword_5008240, &qword_4A427C0);
  qword_5008160 = (__int64)&unk_49DC150;
  v37 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50081B0 = 0x100000000LL;
  dword_500816C &= 0x8000u;
  qword_50081A8 = (__int64)&unk_50081B8;
  word_5008170 = 0;
  qword_5008178 = 0;
  dword_5008168 = v37;
  qword_5008180 = 0;
  qword_5008188 = 0;
  qword_5008190 = 0;
  qword_5008198 = 0;
  qword_50081A0 = 0;
  qword_50081C0 = 0;
  qword_50081C8 = (__int64)&unk_50081E0;
  qword_50081D0 = 1;
  dword_50081D8 = 0;
  byte_50081DC = 1;
  v38 = sub_C57470();
  v39 = (unsigned int)qword_50081B0;
  if ( (unsigned __int64)(unsigned int)qword_50081B0 + 1 > HIDWORD(qword_50081B0) )
  {
    v66 = v38;
    sub_C8D5F0((char *)&unk_50081B8 - 16, &unk_50081B8, (unsigned int)qword_50081B0 + 1LL, 8);
    v39 = (unsigned int)qword_50081B0;
    v38 = v66;
  }
  *(_QWORD *)(qword_50081A8 + 8 * v39) = v38;
  LODWORD(qword_50081B0) = qword_50081B0 + 1;
  qword_50081E8 = 0;
  qword_50081F0 = (__int64)&unk_49D9728;
  qword_50081F8 = 0;
  qword_5008160 = (__int64)&unk_49DBF10;
  qword_5008200 = (__int64)&unk_49DC290;
  qword_5008220 = (__int64)nullsub_24;
  qword_5008218 = (__int64)sub_984050;
  sub_C53080(&qword_5008160, "ext-tsp-max-chain-size", 22);
  LODWORD(qword_50081E8) = 512;
  BYTE4(qword_50081F8) = 1;
  LODWORD(qword_50081F8) = 512;
  qword_5008190 = 37;
  LOBYTE(dword_500816C) = dword_500816C & 0x9F | 0x40;
  qword_5008188 = (__int64)"The maximum size of a chain to create";
  sub_C53130(&qword_5008160);
  __cxa_atexit(sub_984970, &qword_5008160, &qword_4A427C0);
  qword_5008080 = (__int64)&unk_49DC150;
  v40 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50080D0 = 0x100000000LL;
  dword_500808C &= 0x8000u;
  qword_50080C8 = (__int64)&unk_50080D8;
  word_5008090 = 0;
  qword_5008098 = 0;
  dword_5008088 = v40;
  qword_50080A0 = 0;
  qword_50080A8 = 0;
  qword_50080B0 = 0;
  qword_50080B8 = 0;
  qword_50080C0 = 0;
  qword_50080E0 = 0;
  qword_50080E8 = (__int64)&unk_5008100;
  qword_50080F0 = 1;
  dword_50080F8 = 0;
  byte_50080FC = 1;
  v41 = sub_C57470();
  v42 = (unsigned int)qword_50080D0;
  if ( (unsigned __int64)(unsigned int)qword_50080D0 + 1 > HIDWORD(qword_50080D0) )
  {
    v67 = v41;
    sub_C8D5F0((char *)&unk_50080D8 - 16, &unk_50080D8, (unsigned int)qword_50080D0 + 1LL, 8);
    v42 = (unsigned int)qword_50080D0;
    v41 = v67;
  }
  *(_QWORD *)(qword_50080C8 + 8 * v42) = v41;
  LODWORD(qword_50080D0) = qword_50080D0 + 1;
  qword_5008108 = 0;
  qword_5008110 = (__int64)&unk_49D9728;
  qword_5008118 = 0;
  qword_5008080 = (__int64)&unk_49DBF10;
  qword_5008120 = (__int64)&unk_49DC290;
  qword_5008140 = (__int64)nullsub_24;
  qword_5008138 = (__int64)sub_984050;
  sub_C53080(&qword_5008080, "ext-tsp-chain-split-threshold", 29);
  LODWORD(qword_5008108) = 128;
  BYTE4(qword_5008118) = 1;
  LODWORD(qword_5008118) = 128;
  qword_50080B0 = 46;
  LOBYTE(dword_500808C) = dword_500808C & 0x9F | 0x40;
  qword_50080A8 = (__int64)"The maximum size of a chain to apply splitting";
  sub_C53130(&qword_5008080);
  __cxa_atexit(sub_984970, &qword_5008080, &qword_4A427C0);
  qword_5007FA0 = (__int64)&unk_49DC150;
  v43 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5007FF0 = 0x100000000LL;
  dword_5007FAC &= 0x8000u;
  word_5007FB0 = 0;
  qword_5007FE8 = (__int64)&unk_5007FF8;
  qword_5007FB8 = 0;
  dword_5007FA8 = v43;
  qword_5007FC0 = 0;
  qword_5007FC8 = 0;
  qword_5007FD0 = 0;
  qword_5007FD8 = 0;
  qword_5007FE0 = 0;
  qword_5008000 = 0;
  qword_5008008 = (__int64)&unk_5008020;
  qword_5008010 = 1;
  dword_5008018 = 0;
  byte_500801C = 1;
  v44 = sub_C57470();
  v45 = (unsigned int)qword_5007FF0;
  if ( (unsigned __int64)(unsigned int)qword_5007FF0 + 1 > HIDWORD(qword_5007FF0) )
  {
    v68 = v44;
    sub_C8D5F0((char *)&unk_5007FF8 - 16, &unk_5007FF8, (unsigned int)qword_5007FF0 + 1LL, 8);
    v45 = (unsigned int)qword_5007FF0;
    v44 = v68;
  }
  *(_QWORD *)(qword_5007FE8 + 8 * v45) = v44;
  qword_5008030 = (__int64)&unk_49DE5F0;
  qword_5007FA0 = (__int64)&unk_49DE610;
  LODWORD(qword_5007FF0) = qword_5007FF0 + 1;
  byte_5008040 = 0;
  qword_5008048 = (__int64)&unk_49DC2F0;
  qword_5008028 = 0;
  qword_5008068 = (__int64)nullsub_190;
  qword_5008038 = 0;
  qword_5008060 = (__int64)sub_D83E80;
  sub_C53080(&qword_5007FA0, "ext-tsp-max-merge-density-ratio", 31);
  byte_5008040 = 1;
  qword_5008028 = 0x4059000000000000LL;
  qword_5008038 = 0x4059000000000000LL;
  LOBYTE(dword_5007FAC) = dword_5007FAC & 0x9F | 0x40;
  qword_5007FC8 = (__int64)"The maximum ratio between densities of two chains for merging";
  qword_5007FD0 = 61;
  sub_C53130(&qword_5007FA0);
  __cxa_atexit(sub_D84280, &qword_5007FA0, &qword_4A427C0);
  qword_5007EC0 = (__int64)&unk_49DC150;
  v46 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5007F10 = 0x100000000LL;
  dword_5007ECC &= 0x8000u;
  qword_5007F08 = (__int64)&unk_5007F18;
  word_5007ED0 = 0;
  qword_5007ED8 = 0;
  dword_5007EC8 = v46;
  qword_5007EE0 = 0;
  qword_5007EE8 = 0;
  qword_5007EF0 = 0;
  qword_5007EF8 = 0;
  qword_5007F00 = 0;
  qword_5007F20 = 0;
  qword_5007F28 = (__int64)&unk_5007F40;
  qword_5007F30 = 1;
  dword_5007F38 = 0;
  byte_5007F3C = 1;
  v47 = sub_C57470();
  v48 = (unsigned int)qword_5007F10;
  if ( (unsigned __int64)(unsigned int)qword_5007F10 + 1 > HIDWORD(qword_5007F10) )
  {
    v69 = v47;
    sub_C8D5F0((char *)&unk_5007F18 - 16, &unk_5007F18, (unsigned int)qword_5007F10 + 1LL, 8);
    v48 = (unsigned int)qword_5007F10;
    v47 = v69;
  }
  *(_QWORD *)(qword_5007F08 + 8 * v48) = v47;
  LODWORD(qword_5007F10) = qword_5007F10 + 1;
  qword_5007F48 = 0;
  qword_5007F50 = (__int64)&unk_49D9728;
  qword_5007F58 = 0;
  qword_5007EC0 = (__int64)&unk_49DBF10;
  qword_5007F60 = (__int64)&unk_49DC290;
  qword_5007F80 = (__int64)nullsub_24;
  qword_5007F78 = (__int64)sub_984050;
  sub_C53080(&qword_5007EC0, "cdsort-cache-entries", 20);
  qword_5007EF0 = 21;
  LOBYTE(dword_5007ECC) = dword_5007ECC & 0x9F | 0x40;
  qword_5007EE8 = (__int64)"The size of the cache";
  sub_C53130(&qword_5007EC0);
  __cxa_atexit(sub_984970, &qword_5007EC0, &qword_4A427C0);
  qword_5007DE0 = (__int64)&unk_49DC150;
  v49 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5007E5C = 1;
  word_5007DF0 = 0;
  qword_5007E30 = 0x100000000LL;
  dword_5007DEC &= 0x8000u;
  qword_5007E28 = (__int64)&unk_5007E38;
  qword_5007DF8 = 0;
  dword_5007DE8 = v49;
  qword_5007E00 = 0;
  qword_5007E08 = 0;
  qword_5007E10 = 0;
  qword_5007E18 = 0;
  qword_5007E20 = 0;
  qword_5007E40 = 0;
  qword_5007E48 = (__int64)&unk_5007E60;
  qword_5007E50 = 1;
  dword_5007E58 = 0;
  v50 = sub_C57470();
  v51 = (unsigned int)qword_5007E30;
  if ( (unsigned __int64)(unsigned int)qword_5007E30 + 1 > HIDWORD(qword_5007E30) )
  {
    v70 = v50;
    sub_C8D5F0((char *)&unk_5007E38 - 16, &unk_5007E38, (unsigned int)qword_5007E30 + 1LL, 8);
    v51 = (unsigned int)qword_5007E30;
    v50 = v70;
  }
  *(_QWORD *)(qword_5007E28 + 8 * v51) = v50;
  LODWORD(qword_5007E30) = qword_5007E30 + 1;
  qword_5007E68 = 0;
  qword_5007E70 = (__int64)&unk_49D9728;
  qword_5007E78 = 0;
  qword_5007DE0 = (__int64)&unk_49DBF10;
  qword_5007E80 = (__int64)&unk_49DC290;
  qword_5007EA0 = (__int64)nullsub_24;
  qword_5007E98 = (__int64)sub_984050;
  sub_C53080(&qword_5007DE0, "cdsort-cache-size", 17);
  qword_5007E10 = 31;
  LOBYTE(dword_5007DEC) = dword_5007DEC & 0x9F | 0x40;
  qword_5007E08 = (__int64)"The size of a line in the cache";
  sub_C53130(&qword_5007DE0);
  __cxa_atexit(sub_984970, &qword_5007DE0, &qword_4A427C0);
  qword_5007D00 = (__int64)&unk_49DC150;
  v52 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5007D50 = 0x100000000LL;
  dword_5007D0C &= 0x8000u;
  word_5007D10 = 0;
  qword_5007D48 = (__int64)&unk_5007D58;
  qword_5007D18 = 0;
  dword_5007D08 = v52;
  qword_5007D20 = 0;
  qword_5007D28 = 0;
  qword_5007D30 = 0;
  qword_5007D38 = 0;
  qword_5007D40 = 0;
  qword_5007D60 = 0;
  qword_5007D68 = (__int64)&unk_5007D80;
  qword_5007D70 = 1;
  dword_5007D78 = 0;
  byte_5007D7C = 1;
  v53 = sub_C57470();
  v54 = (unsigned int)qword_5007D50;
  if ( (unsigned __int64)(unsigned int)qword_5007D50 + 1 > HIDWORD(qword_5007D50) )
  {
    v71 = v53;
    sub_C8D5F0((char *)&unk_5007D58 - 16, &unk_5007D58, (unsigned int)qword_5007D50 + 1LL, 8);
    v54 = (unsigned int)qword_5007D50;
    v53 = v71;
  }
  *(_QWORD *)(qword_5007D48 + 8 * v54) = v53;
  LODWORD(qword_5007D50) = qword_5007D50 + 1;
  qword_5007D88 = 0;
  qword_5007D90 = (__int64)&unk_49D9728;
  qword_5007D98 = 0;
  qword_5007D00 = (__int64)&unk_49DBF10;
  qword_5007DA0 = (__int64)&unk_49DC290;
  qword_5007DC0 = (__int64)nullsub_24;
  qword_5007DB8 = (__int64)sub_984050;
  sub_C53080(&qword_5007D00, "cdsort-max-chain-size", 21);
  qword_5007D30 = 37;
  LOBYTE(dword_5007D0C) = dword_5007D0C & 0x9F | 0x40;
  qword_5007D28 = (__int64)"The maximum size of a chain to create";
  sub_C53130(&qword_5007D00);
  __cxa_atexit(sub_984970, &qword_5007D00, &qword_4A427C0);
  qword_5007C20 = (__int64)&unk_49DC150;
  v55 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5007C9C = 1;
  qword_5007C70 = 0x100000000LL;
  dword_5007C2C &= 0x8000u;
  qword_5007C68 = (__int64)&unk_5007C78;
  qword_5007C38 = 0;
  qword_5007C40 = 0;
  dword_5007C28 = v55;
  word_5007C30 = 0;
  qword_5007C48 = 0;
  qword_5007C50 = 0;
  qword_5007C58 = 0;
  qword_5007C60 = 0;
  qword_5007C80 = 0;
  qword_5007C88 = (__int64)&unk_5007CA0;
  qword_5007C90 = 1;
  dword_5007C98 = 0;
  v56 = sub_C57470();
  v57 = (unsigned int)qword_5007C70;
  v58 = (unsigned int)qword_5007C70 + 1LL;
  if ( v58 > HIDWORD(qword_5007C70) )
  {
    sub_C8D5F0((char *)&unk_5007C78 - 16, &unk_5007C78, v58, 8);
    v57 = (unsigned int)qword_5007C70;
  }
  *(_QWORD *)(qword_5007C68 + 8 * v57) = v56;
  qword_5007CB0 = (__int64)&unk_49DE5F0;
  qword_5007C20 = (__int64)&unk_49DE610;
  LODWORD(qword_5007C70) = qword_5007C70 + 1;
  byte_5007CC0 = 0;
  qword_5007CC8 = (__int64)&unk_49DC2F0;
  qword_5007CA8 = 0;
  qword_5007CE8 = (__int64)nullsub_190;
  qword_5007CB8 = 0;
  qword_5007CE0 = (__int64)sub_D83E80;
  sub_C53080(&qword_5007C20, "cdsort-distance-power", 21);
  qword_5007C50 = 50;
  LOBYTE(dword_5007C2C) = dword_5007C2C & 0x9F | 0x40;
  qword_5007C48 = (__int64)"The power exponent for the distance-based locality";
  sub_C53130(&qword_5007C20);
  __cxa_atexit(sub_D84280, &qword_5007C20, &qword_4A427C0);
  qword_5007B40 = (__int64)&unk_49DC150;
  v59 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_5007B4C &= 0x8000u;
  word_5007B50 = 0;
  qword_5007B90 = 0x100000000LL;
  qword_5007B88 = (__int64)&unk_5007B98;
  qword_5007B58 = 0;
  qword_5007B60 = 0;
  dword_5007B48 = v59;
  qword_5007B68 = 0;
  qword_5007B70 = 0;
  qword_5007B78 = 0;
  qword_5007B80 = 0;
  qword_5007BA0 = 0;
  qword_5007BA8 = (__int64)&unk_5007BC0;
  qword_5007BB0 = 1;
  dword_5007BB8 = 0;
  byte_5007BBC = 1;
  v60 = sub_C57470();
  v61 = (unsigned int)qword_5007B90;
  v62 = (unsigned int)qword_5007B90 + 1LL;
  if ( v62 > HIDWORD(qword_5007B90) )
  {
    sub_C8D5F0((char *)&unk_5007B98 - 16, &unk_5007B98, v62, 8);
    v61 = (unsigned int)qword_5007B90;
  }
  *(_QWORD *)(qword_5007B88 + 8 * v61) = v60;
  qword_5007BD0 = (__int64)&unk_49DE5F0;
  qword_5007B40 = (__int64)&unk_49DE610;
  LODWORD(qword_5007B90) = qword_5007B90 + 1;
  byte_5007BE0 = 0;
  qword_5007BE8 = (__int64)&unk_49DC2F0;
  qword_5007BC8 = 0;
  qword_5007C08 = (__int64)nullsub_190;
  qword_5007BD8 = 0;
  qword_5007C00 = (__int64)sub_D83E80;
  sub_C53080(&qword_5007B40, "cdsort-frequency-scale", 22);
  qword_5007B70 = 49;
  LOBYTE(dword_5007B4C) = dword_5007B4C & 0x9F | 0x40;
  qword_5007B68 = (__int64)"The scale factor for the frequency-based locality";
  sub_C53130(&qword_5007B40);
  __cxa_atexit(sub_D84280, &qword_5007B40, &qword_4A427C0);
  qword_5007B10 = 0;
  qword_5007B18 = 0;
  qword_5007B20 = 0;
  return __cxa_atexit(sub_29B86A0, &qword_5007B10, &qword_4A427C0);
}

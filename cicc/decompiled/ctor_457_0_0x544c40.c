// Function: ctor_457_0
// Address: 0x544c40
//
int ctor_457_0()
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
  __int64 v9; // r12
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  int v12; // edx
  __int64 v13; // r12
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rdx
  int v19; // edx
  __int64 v20; // rax
  __int64 v21; // rdx
  int v22; // edx
  __int64 v23; // rax
  __int64 v24; // rdx
  int v25; // edx
  __int64 v26; // rax
  __int64 v27; // rdx
  int v28; // edx
  __int64 v29; // rax
  __int64 v30; // rdx
  int v31; // edx
  __int64 v32; // rax
  __int64 v33; // rdx
  int v34; // edx
  __int64 v35; // rbx
  __int64 v36; // rax
  __int64 v38; // [rsp+8h] [rbp-38h]
  __int64 v39; // [rsp+8h] [rbp-38h]
  __int64 v40; // [rsp+8h] [rbp-38h]
  __int64 v41; // [rsp+8h] [rbp-38h]
  __int64 v42; // [rsp+8h] [rbp-38h]
  __int64 v43; // [rsp+8h] [rbp-38h]

  qword_4FFE620 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FFE62C &= 0x8000u;
  word_4FFE630 = 0;
  qword_4FFE670 = 0x100000000LL;
  qword_4FFE638 = 0;
  qword_4FFE640 = 0;
  qword_4FFE648 = 0;
  dword_4FFE628 = v0;
  qword_4FFE650 = 0;
  qword_4FFE658 = 0;
  qword_4FFE660 = 0;
  qword_4FFE668 = (__int64)&unk_4FFE678;
  qword_4FFE680 = 0;
  qword_4FFE688 = (__int64)&unk_4FFE6A0;
  qword_4FFE690 = 1;
  dword_4FFE698 = 0;
  byte_4FFE69C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FFE670;
  v3 = (unsigned int)qword_4FFE670 + 1LL;
  if ( v3 > HIDWORD(qword_4FFE670) )
  {
    sub_C8D5F0((char *)&unk_4FFE678 - 16, &unk_4FFE678, v3, 8);
    v2 = (unsigned int)qword_4FFE670;
  }
  *(_QWORD *)(qword_4FFE668 + 8 * v2) = v1;
  LODWORD(qword_4FFE670) = qword_4FFE670 + 1;
  qword_4FFE6A8 = 0;
  qword_4FFE6B0 = (__int64)&unk_49D9748;
  qword_4FFE6B8 = 0;
  qword_4FFE620 = (__int64)&unk_49DC090;
  qword_4FFE6C0 = (__int64)&unk_49DC1D0;
  qword_4FFE6E0 = (__int64)nullsub_23;
  qword_4FFE6D8 = (__int64)sub_984030;
  sub_C53080(&qword_4FFE620, "disable-licm-promotion", 22);
  LOWORD(qword_4FFE6B8) = 256;
  LOBYTE(qword_4FFE6A8) = 0;
  qword_4FFE650 = 37;
  LOBYTE(dword_4FFE62C) = dword_4FFE62C & 0x9F | 0x20;
  qword_4FFE648 = (__int64)"Disable memory promotion in LICM pass";
  sub_C53130(&qword_4FFE620);
  __cxa_atexit(sub_984900, &qword_4FFE620, &qword_4A427C0);
  qword_4FFE540 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FFE54C &= 0x8000u;
  word_4FFE550 = 0;
  qword_4FFE590 = 0x100000000LL;
  qword_4FFE558 = 0;
  qword_4FFE560 = 0;
  qword_4FFE568 = 0;
  dword_4FFE548 = v4;
  qword_4FFE570 = 0;
  qword_4FFE578 = 0;
  qword_4FFE580 = 0;
  qword_4FFE588 = (__int64)&unk_4FFE598;
  qword_4FFE5A0 = 0;
  qword_4FFE5A8 = (__int64)&unk_4FFE5C0;
  qword_4FFE5B0 = 1;
  dword_4FFE5B8 = 0;
  byte_4FFE5BC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FFE590;
  v7 = (unsigned int)qword_4FFE590 + 1LL;
  if ( v7 > HIDWORD(qword_4FFE590) )
  {
    sub_C8D5F0((char *)&unk_4FFE598 - 16, &unk_4FFE598, v7, 8);
    v6 = (unsigned int)qword_4FFE590;
  }
  *(_QWORD *)(qword_4FFE588 + 8 * v6) = v5;
  LODWORD(qword_4FFE590) = qword_4FFE590 + 1;
  qword_4FFE5C8 = 0;
  qword_4FFE5D0 = (__int64)&unk_49D9748;
  qword_4FFE5D8 = 0;
  qword_4FFE540 = (__int64)&unk_49DC090;
  qword_4FFE5E0 = (__int64)&unk_49DC1D0;
  qword_4FFE600 = (__int64)nullsub_23;
  qword_4FFE5F8 = (__int64)sub_984030;
  sub_C53080(&qword_4FFE540, "licm-control-flow-hoisting", 26);
  LOWORD(qword_4FFE5D8) = 256;
  LOBYTE(qword_4FFE5C8) = 0;
  qword_4FFE570 = 46;
  LOBYTE(dword_4FFE54C) = dword_4FFE54C & 0x9F | 0x20;
  qword_4FFE568 = (__int64)"Enable control flow (and PHI) hoisting in LICM";
  sub_C53130(&qword_4FFE540);
  __cxa_atexit(sub_984900, &qword_4FFE540, &qword_4A427C0);
  qword_4FFE460 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFE4B0 = 0x100000000LL;
  dword_4FFE46C &= 0x8000u;
  word_4FFE470 = 0;
  qword_4FFE478 = 0;
  qword_4FFE480 = 0;
  dword_4FFE468 = v8;
  qword_4FFE488 = 0;
  qword_4FFE490 = 0;
  qword_4FFE498 = 0;
  qword_4FFE4A0 = 0;
  qword_4FFE4A8 = (__int64)&unk_4FFE4B8;
  qword_4FFE4C0 = 0;
  qword_4FFE4C8 = (__int64)&unk_4FFE4E0;
  qword_4FFE4D0 = 1;
  dword_4FFE4D8 = 0;
  byte_4FFE4DC = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4FFE4B0;
  v11 = (unsigned int)qword_4FFE4B0 + 1LL;
  if ( v11 > HIDWORD(qword_4FFE4B0) )
  {
    sub_C8D5F0((char *)&unk_4FFE4B8 - 16, &unk_4FFE4B8, v11, 8);
    v10 = (unsigned int)qword_4FFE4B0;
  }
  *(_QWORD *)(qword_4FFE4A8 + 8 * v10) = v9;
  LODWORD(qword_4FFE4B0) = qword_4FFE4B0 + 1;
  qword_4FFE4E8 = 0;
  qword_4FFE4F0 = (__int64)&unk_49D9748;
  qword_4FFE4F8 = 0;
  qword_4FFE460 = (__int64)&unk_49DC090;
  qword_4FFE500 = (__int64)&unk_49DC1D0;
  qword_4FFE520 = (__int64)nullsub_23;
  qword_4FFE518 = (__int64)sub_984030;
  sub_C53080(&qword_4FFE460, "licm-force-thread-model-single", 30);
  LOWORD(qword_4FFE4F8) = 256;
  LOBYTE(qword_4FFE4E8) = 0;
  qword_4FFE490 = 38;
  LOBYTE(dword_4FFE46C) = dword_4FFE46C & 0x9F | 0x20;
  qword_4FFE488 = (__int64)"Force thread model single in LICM pass";
  sub_C53130(&qword_4FFE460);
  __cxa_atexit(sub_984900, &qword_4FFE460, &qword_4A427C0);
  qword_4FFE380 = (__int64)&unk_49DC150;
  v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFE3D0 = 0x100000000LL;
  word_4FFE390 = 0;
  dword_4FFE38C &= 0x8000u;
  qword_4FFE398 = 0;
  qword_4FFE3A0 = 0;
  dword_4FFE388 = v12;
  qword_4FFE3A8 = 0;
  qword_4FFE3B0 = 0;
  qword_4FFE3B8 = 0;
  qword_4FFE3C0 = 0;
  qword_4FFE3C8 = (__int64)&unk_4FFE3D8;
  qword_4FFE3E0 = 0;
  qword_4FFE3E8 = (__int64)&unk_4FFE400;
  qword_4FFE3F0 = 1;
  dword_4FFE3F8 = 0;
  byte_4FFE3FC = 1;
  v13 = sub_C57470();
  v14 = (unsigned int)qword_4FFE3D0;
  v15 = (unsigned int)qword_4FFE3D0 + 1LL;
  if ( v15 > HIDWORD(qword_4FFE3D0) )
  {
    sub_C8D5F0((char *)&unk_4FFE3D8 - 16, &unk_4FFE3D8, v15, 8);
    v14 = (unsigned int)qword_4FFE3D0;
  }
  *(_QWORD *)(qword_4FFE3C8 + 8 * v14) = v13;
  qword_4FFE410 = (__int64)&unk_49D9728;
  qword_4FFE380 = (__int64)&unk_49DBF10;
  LODWORD(qword_4FFE3D0) = qword_4FFE3D0 + 1;
  qword_4FFE408 = 0;
  qword_4FFE420 = (__int64)&unk_49DC290;
  qword_4FFE418 = 0;
  qword_4FFE440 = (__int64)nullsub_24;
  qword_4FFE438 = (__int64)sub_984050;
  sub_C53080(&qword_4FFE380, "licm-max-num-uses-traversed", 27);
  LODWORD(qword_4FFE408) = 8;
  BYTE4(qword_4FFE418) = 1;
  LODWORD(qword_4FFE418) = 8;
  qword_4FFE3B0 = 96;
  LOBYTE(dword_4FFE38C) = dword_4FFE38C & 0x9F | 0x20;
  qword_4FFE3A8 = (__int64)"Max num uses visited for identifying load invariance in loop using invariant start (default = 8)";
  sub_C53130(&qword_4FFE380);
  __cxa_atexit(sub_984970, &qword_4FFE380, &qword_4A427C0);
  qword_4FFE2A0 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFE2F0 = 0x100000000LL;
  dword_4FFE2AC &= 0x8000u;
  word_4FFE2B0 = 0;
  qword_4FFE2B8 = 0;
  qword_4FFE2C0 = 0;
  dword_4FFE2A8 = v16;
  qword_4FFE2C8 = 0;
  qword_4FFE2D0 = 0;
  qword_4FFE2D8 = 0;
  qword_4FFE2E0 = 0;
  qword_4FFE2E8 = (__int64)&unk_4FFE2F8;
  qword_4FFE300 = 0;
  qword_4FFE308 = (__int64)&unk_4FFE320;
  qword_4FFE310 = 1;
  dword_4FFE318 = 0;
  byte_4FFE31C = 1;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_4FFE2F0;
  if ( (unsigned __int64)(unsigned int)qword_4FFE2F0 + 1 > HIDWORD(qword_4FFE2F0) )
  {
    v38 = v17;
    sub_C8D5F0((char *)&unk_4FFE2F8 - 16, &unk_4FFE2F8, (unsigned int)qword_4FFE2F0 + 1LL, 8);
    v18 = (unsigned int)qword_4FFE2F0;
    v17 = v38;
  }
  *(_QWORD *)(qword_4FFE2E8 + 8 * v18) = v17;
  qword_4FFE330 = (__int64)&unk_49D9728;
  qword_4FFE2A0 = (__int64)&unk_49DBF10;
  LODWORD(qword_4FFE2F0) = qword_4FFE2F0 + 1;
  qword_4FFE328 = 0;
  qword_4FFE340 = (__int64)&unk_49DC290;
  qword_4FFE338 = 0;
  qword_4FFE360 = (__int64)nullsub_24;
  qword_4FFE358 = (__int64)sub_984050;
  sub_C53080(&qword_4FFE2A0, "licm-max-num-fp-reassociations", 30);
  qword_4FFE2C8 = (__int64)"Set upper limit for the number of transformations performed during a single round of hoisting"
                           " the reassociated expressions.";
  LODWORD(qword_4FFE328) = 5;
  BYTE4(qword_4FFE338) = 1;
  LODWORD(qword_4FFE338) = 5;
  LOBYTE(dword_4FFE2AC) = dword_4FFE2AC & 0x9F | 0x20;
  qword_4FFE2D0 = 123;
  sub_C53130(&qword_4FFE2A0);
  __cxa_atexit(sub_984970, &qword_4FFE2A0, &qword_4A427C0);
  qword_4FFE1C0 = (__int64)&unk_49DC150;
  v19 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFE210 = 0x100000000LL;
  dword_4FFE1CC &= 0x8000u;
  qword_4FFE208 = (__int64)&unk_4FFE218;
  word_4FFE1D0 = 0;
  qword_4FFE1D8 = 0;
  dword_4FFE1C8 = v19;
  qword_4FFE1E0 = 0;
  qword_4FFE1E8 = 0;
  qword_4FFE1F0 = 0;
  qword_4FFE1F8 = 0;
  qword_4FFE200 = 0;
  qword_4FFE220 = 0;
  qword_4FFE228 = (__int64)&unk_4FFE240;
  qword_4FFE230 = 1;
  dword_4FFE238 = 0;
  byte_4FFE23C = 1;
  v20 = sub_C57470();
  v21 = (unsigned int)qword_4FFE210;
  if ( (unsigned __int64)(unsigned int)qword_4FFE210 + 1 > HIDWORD(qword_4FFE210) )
  {
    v39 = v20;
    sub_C8D5F0((char *)&unk_4FFE218 - 16, &unk_4FFE218, (unsigned int)qword_4FFE210 + 1LL, 8);
    v21 = (unsigned int)qword_4FFE210;
    v20 = v39;
  }
  *(_QWORD *)(qword_4FFE208 + 8 * v21) = v20;
  qword_4FFE250 = (__int64)&unk_49D9728;
  qword_4FFE1C0 = (__int64)&unk_49DBF10;
  LODWORD(qword_4FFE210) = qword_4FFE210 + 1;
  qword_4FFE248 = 0;
  qword_4FFE260 = (__int64)&unk_49DC290;
  qword_4FFE258 = 0;
  qword_4FFE280 = (__int64)nullsub_24;
  qword_4FFE278 = (__int64)sub_984050;
  sub_C53080(&qword_4FFE1C0, "licm-hoist-bo-association-user-limit", 36);
  LODWORD(qword_4FFE248) = 1;
  BYTE4(qword_4FFE258) = 1;
  LODWORD(qword_4FFE258) = 1;
  qword_4FFE1F0 = 99;
  LOBYTE(dword_4FFE1CC) = dword_4FFE1CC & 0x9F | 0x20;
  qword_4FFE1E8 = (__int64)"Limit the number of users of the variant operand when reassociating a binary operator for hoisting.";
  sub_C53130(&qword_4FFE1C0);
  __cxa_atexit(sub_984970, &qword_4FFE1C0, &qword_4A427C0);
  qword_4FFE0E0 = (__int64)&unk_49DC150;
  v22 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFE130 = 0x100000000LL;
  dword_4FFE0EC &= 0x8000u;
  word_4FFE0F0 = 0;
  qword_4FFE128 = (__int64)&unk_4FFE138;
  qword_4FFE0F8 = 0;
  dword_4FFE0E8 = v22;
  qword_4FFE100 = 0;
  qword_4FFE108 = 0;
  qword_4FFE110 = 0;
  qword_4FFE118 = 0;
  qword_4FFE120 = 0;
  qword_4FFE140 = 0;
  qword_4FFE148 = (__int64)&unk_4FFE160;
  qword_4FFE150 = 1;
  dword_4FFE158 = 0;
  byte_4FFE15C = 1;
  v23 = sub_C57470();
  v24 = (unsigned int)qword_4FFE130;
  if ( (unsigned __int64)(unsigned int)qword_4FFE130 + 1 > HIDWORD(qword_4FFE130) )
  {
    v40 = v23;
    sub_C8D5F0((char *)&unk_4FFE138 - 16, &unk_4FFE138, (unsigned int)qword_4FFE130 + 1LL, 8);
    v24 = (unsigned int)qword_4FFE130;
    v23 = v40;
  }
  *(_QWORD *)(qword_4FFE128 + 8 * v24) = v23;
  LODWORD(qword_4FFE130) = qword_4FFE130 + 1;
  qword_4FFE168 = 0;
  qword_4FFE170 = (__int64)&unk_49D9748;
  qword_4FFE178 = 0;
  qword_4FFE0E0 = (__int64)&unk_49DC090;
  qword_4FFE180 = (__int64)&unk_49DC1D0;
  qword_4FFE1A0 = (__int64)nullsub_23;
  qword_4FFE198 = (__int64)sub_984030;
  sub_C53080(&qword_4FFE0E0, "licm-skip-unrolled-loops", 24);
  LOWORD(qword_4FFE178) = 256;
  LOBYTE(qword_4FFE168) = 0;
  qword_4FFE110 = 52;
  LOBYTE(dword_4FFE0EC) = dword_4FFE0EC & 0x9F | 0x20;
  qword_4FFE108 = (__int64)"Skip LICM on loops that are due to be fully unrolled";
  sub_C53130(&qword_4FFE0E0);
  __cxa_atexit(sub_984900, &qword_4FFE0E0, &qword_4A427C0);
  qword_4FFE000 = (__int64)&unk_49DC150;
  v25 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFE050 = 0x100000000LL;
  dword_4FFE00C &= 0x8000u;
  qword_4FFE048 = (__int64)&unk_4FFE058;
  word_4FFE010 = 0;
  qword_4FFE018 = 0;
  dword_4FFE008 = v25;
  qword_4FFE020 = 0;
  qword_4FFE028 = 0;
  qword_4FFE030 = 0;
  qword_4FFE038 = 0;
  qword_4FFE040 = 0;
  qword_4FFE060 = 0;
  qword_4FFE068 = (__int64)&unk_4FFE080;
  qword_4FFE070 = 1;
  dword_4FFE078 = 0;
  byte_4FFE07C = 1;
  v26 = sub_C57470();
  v27 = (unsigned int)qword_4FFE050;
  if ( (unsigned __int64)(unsigned int)qword_4FFE050 + 1 > HIDWORD(qword_4FFE050) )
  {
    v41 = v26;
    sub_C8D5F0((char *)&unk_4FFE058 - 16, &unk_4FFE058, (unsigned int)qword_4FFE050 + 1LL, 8);
    v27 = (unsigned int)qword_4FFE050;
    v26 = v41;
  }
  *(_QWORD *)(qword_4FFE048 + 8 * v27) = v26;
  qword_4FFE090 = (__int64)&unk_49D9728;
  qword_4FFE000 = (__int64)&unk_49DBF10;
  LODWORD(qword_4FFE050) = qword_4FFE050 + 1;
  qword_4FFE088 = 0;
  qword_4FFE0A0 = (__int64)&unk_49DC290;
  qword_4FFE098 = 0;
  qword_4FFE0C0 = (__int64)nullsub_24;
  qword_4FFE0B8 = (__int64)sub_984050;
  sub_C53080(&qword_4FFE000, "licm-insn-limit", 15);
  LODWORD(qword_4FFE088) = 500;
  BYTE4(qword_4FFE098) = 1;
  LODWORD(qword_4FFE098) = 500;
  qword_4FFE030 = 40;
  LOBYTE(dword_4FFE00C) = dword_4FFE00C & 0x9F | 0x20;
  qword_4FFE028 = (__int64)"Control the loop-size threshold for LICM";
  sub_C53130(&qword_4FFE000);
  __cxa_atexit(sub_984970, &qword_4FFE000, &qword_4A427C0);
  qword_4FFDF20 = (__int64)&unk_49DC150;
  v28 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFDF70 = 0x100000000LL;
  dword_4FFDF2C &= 0x8000u;
  word_4FFDF30 = 0;
  qword_4FFDF68 = (__int64)&unk_4FFDF78;
  qword_4FFDF38 = 0;
  dword_4FFDF28 = v28;
  qword_4FFDF40 = 0;
  qword_4FFDF48 = 0;
  qword_4FFDF50 = 0;
  qword_4FFDF58 = 0;
  qword_4FFDF60 = 0;
  qword_4FFDF80 = 0;
  qword_4FFDF88 = (__int64)&unk_4FFDFA0;
  qword_4FFDF90 = 1;
  dword_4FFDF98 = 0;
  byte_4FFDF9C = 1;
  v29 = sub_C57470();
  v30 = (unsigned int)qword_4FFDF70;
  if ( (unsigned __int64)(unsigned int)qword_4FFDF70 + 1 > HIDWORD(qword_4FFDF70) )
  {
    v42 = v29;
    sub_C8D5F0((char *)&unk_4FFDF78 - 16, &unk_4FFDF78, (unsigned int)qword_4FFDF70 + 1LL, 8);
    v30 = (unsigned int)qword_4FFDF70;
    v29 = v42;
  }
  *(_QWORD *)(qword_4FFDF68 + 8 * v30) = v29;
  qword_4FFDFB0 = (__int64)&unk_49D9728;
  qword_4FFDF20 = (__int64)&unk_49DBF10;
  LODWORD(qword_4FFDF70) = qword_4FFDF70 + 1;
  qword_4FFDFA8 = 0;
  qword_4FFDFC0 = (__int64)&unk_49DC290;
  qword_4FFDFB8 = 0;
  qword_4FFDFE0 = (__int64)nullsub_24;
  qword_4FFDFD8 = (__int64)sub_984050;
  sub_C53080(&qword_4FFDF20, "licm-max-num-int-reassociations", 31);
  qword_4FFDF48 = (__int64)"Set upper limit for the number of transformations performed during a single round of hoisting"
                           " the reassociated expressions.";
  LODWORD(qword_4FFDFA8) = 5;
  BYTE4(qword_4FFDFB8) = 1;
  LODWORD(qword_4FFDFB8) = 5;
  LOBYTE(dword_4FFDF2C) = dword_4FFDF2C & 0x9F | 0x20;
  qword_4FFDF50 = 123;
  sub_C53130(&qword_4FFDF20);
  __cxa_atexit(sub_984970, &qword_4FFDF20, &qword_4A427C0);
  qword_4FFDE40 = &unk_49DC150;
  v31 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4FFDE4C = word_4FFDE4C & 0x8000;
  unk_4FFDE48 = v31;
  qword_4FFDE88[1] = 0x100000000LL;
  unk_4FFDE50 = 0;
  qword_4FFDE88[0] = &qword_4FFDE88[2];
  unk_4FFDE58 = 0;
  unk_4FFDE60 = 0;
  unk_4FFDE68 = 0;
  unk_4FFDE70 = 0;
  unk_4FFDE78 = 0;
  unk_4FFDE80 = 0;
  qword_4FFDE88[3] = 0;
  qword_4FFDE88[4] = &qword_4FFDE88[7];
  qword_4FFDE88[5] = 1;
  LODWORD(qword_4FFDE88[6]) = 0;
  BYTE4(qword_4FFDE88[6]) = 1;
  v32 = sub_C57470();
  v33 = LODWORD(qword_4FFDE88[1]);
  if ( (unsigned __int64)LODWORD(qword_4FFDE88[1]) + 1 > HIDWORD(qword_4FFDE88[1]) )
  {
    v43 = v32;
    sub_C8D5F0(qword_4FFDE88, &qword_4FFDE88[2], LODWORD(qword_4FFDE88[1]) + 1LL, 8);
    v33 = LODWORD(qword_4FFDE88[1]);
    v32 = v43;
  }
  *(_QWORD *)(qword_4FFDE88[0] + 8 * v33) = v32;
  qword_4FFDE88[9] = &unk_49D9728;
  qword_4FFDE40 = &unk_49DBF10;
  ++LODWORD(qword_4FFDE88[1]);
  qword_4FFDE88[8] = 0;
  qword_4FFDE88[11] = &unk_49DC290;
  qword_4FFDE88[10] = 0;
  qword_4FFDE88[15] = nullsub_24;
  qword_4FFDE88[14] = sub_984050;
  sub_C53080(&qword_4FFDE40, "licm-mssa-optimization-cap", 26);
  BYTE4(qword_4FFDE88[10]) = 1;
  LODWORD(qword_4FFDE88[8]) = 100;
  unk_4FFDE70 = 118;
  LODWORD(qword_4FFDE88[10]) = 100;
  LOBYTE(word_4FFDE4C) = word_4FFDE4C & 0x9F | 0x20;
  unk_4FFDE68 = "Enable imprecision in LICM in pathological cases, in exchange for faster compile. Caps the MemorySSA clobbering calls.";
  sub_C53130(&qword_4FFDE40);
  __cxa_atexit(sub_984970, &qword_4FFDE40, &qword_4A427C0);
  qword_4FFDD60 = &unk_49DC150;
  v34 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4FFDD6C = word_4FFDD6C & 0x8000;
  unk_4FFDD70 = 0;
  qword_4FFDDA8[1] = 0x100000000LL;
  unk_4FFDD68 = v34;
  qword_4FFDDA8[0] = &qword_4FFDDA8[2];
  unk_4FFDD78 = 0;
  unk_4FFDD80 = 0;
  unk_4FFDD88 = 0;
  unk_4FFDD90 = 0;
  unk_4FFDD98 = 0;
  unk_4FFDDA0 = 0;
  qword_4FFDDA8[3] = 0;
  qword_4FFDDA8[4] = &qword_4FFDDA8[7];
  qword_4FFDDA8[5] = 1;
  LODWORD(qword_4FFDDA8[6]) = 0;
  BYTE4(qword_4FFDDA8[6]) = 1;
  v35 = sub_C57470();
  v36 = LODWORD(qword_4FFDDA8[1]);
  if ( (unsigned __int64)LODWORD(qword_4FFDDA8[1]) + 1 > HIDWORD(qword_4FFDDA8[1]) )
  {
    sub_C8D5F0(qword_4FFDDA8, &qword_4FFDDA8[2], LODWORD(qword_4FFDDA8[1]) + 1LL, 8);
    v36 = LODWORD(qword_4FFDDA8[1]);
  }
  *(_QWORD *)(qword_4FFDDA8[0] + 8 * v36) = v35;
  qword_4FFDDA8[9] = &unk_49D9728;
  qword_4FFDD60 = &unk_49DBF10;
  ++LODWORD(qword_4FFDDA8[1]);
  qword_4FFDDA8[8] = 0;
  qword_4FFDDA8[11] = &unk_49DC290;
  qword_4FFDDA8[10] = 0;
  qword_4FFDDA8[15] = nullsub_24;
  qword_4FFDDA8[14] = sub_984050;
  sub_C53080(&qword_4FFDD60, "licm-mssa-max-acc-promotion", 27);
  BYTE4(qword_4FFDDA8[10]) = 1;
  LODWORD(qword_4FFDDA8[8]) = 250;
  unk_4FFDD90 = 212;
  LODWORD(qword_4FFDDA8[10]) = 250;
  LOBYTE(word_4FFDD6C) = word_4FFDD6C & 0x9F | 0x20;
  unk_4FFDD88 = "[LICM & MemorySSA] When MSSA in LICM is disabled, this has no effect. When MSSA in LICM is enabled, then"
                " this is the maximum number of accesses allowed to be present in a loop in order to enable memory promotion.";
  sub_C53130(&qword_4FFDD60);
  return __cxa_atexit(sub_984970, &qword_4FFDD60, &qword_4A427C0);
}

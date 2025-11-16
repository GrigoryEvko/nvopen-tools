// Function: ctor_068_0
// Address: 0x4971a0
//
int ctor_068_0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  int v2; // edx
  __int64 v3; // rax
  __int64 v4; // rdx
  int v5; // edx
  __int64 v6; // rax
  __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // rbx
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  int v15; // edx
  __int64 v16; // rbx
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v19; // rax
  int v20; // edx
  __int64 v21; // rbx
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
  int v32; // edx
  __int64 v33; // rbx
  __int64 v34; // rax
  unsigned __int64 v35; // rdx
  int v36; // edx
  __int64 v37; // rax
  __int64 v38; // rdx
  int v39; // edx
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v43; // [rsp+0h] [rbp-B0h]
  __int64 v44; // [rsp+8h] [rbp-A8h]
  __int64 v45; // [rsp+8h] [rbp-A8h]
  __int64 v46; // [rsp+18h] [rbp-98h]
  __int64 v47; // [rsp+18h] [rbp-98h]
  char v48; // [rsp+27h] [rbp-89h] BYREF
  int v49; // [rsp+28h] [rbp-88h] BYREF
  int v50; // [rsp+2Ch] [rbp-84h] BYREF
  _QWORD v51[2]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD v52[2]; // [rsp+40h] [rbp-70h] BYREF
  const char *v53; // [rsp+50h] [rbp-60h] BYREF
  __int64 v54; // [rsp+58h] [rbp-58h]
  _QWORD v55[2]; // [rsp+60h] [rbp-50h] BYREF
  char v56; // [rsp+70h] [rbp-40h]
  char v57; // [rsp+71h] [rbp-3Fh]

  v0 = sub_C60B10();
  v53 = (const char *)v55;
  v1 = v0;
  sub_F06C10(&v53, "Controls which instructions are visited");
  v51[0] = v52;
  sub_F06C10(v51, "instcombine-visit");
  sub_CF9810(v1, v51, &v53);
  if ( (_QWORD *)v51[0] != v52 )
    j_j___libc_free_0(v51[0], v52[0] + 1LL);
  if ( v53 != (const char *)v55 )
    j_j___libc_free_0(v53, v55[0] + 1LL);
  v53 = "Disable ADD to OR transformations";
  v51[0] = &v48;
  v54 = 33;
  v50 = 1;
  v49 = 1;
  v48 = 1;
  sub_F11640(&unk_4F8BAE0, "disable-add-to-or", v51, &v49, &v50, &v53);
  __cxa_atexit(sub_984900, &unk_4F8BAE0, &qword_4A427C0);
  qword_4F8BA00 = &unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8BA0C = word_4F8BA0C & 0x8000;
  qword_4F8BA48[1] = 0x100000000LL;
  unk_4F8BA08 = v2;
  unk_4F8BA10 = 0;
  qword_4F8BA48[0] = &qword_4F8BA48[2];
  unk_4F8BA18 = 0;
  unk_4F8BA20 = 0;
  unk_4F8BA28 = 0;
  unk_4F8BA30 = 0;
  unk_4F8BA38 = 0;
  unk_4F8BA40 = 0;
  qword_4F8BA48[3] = 0;
  qword_4F8BA48[4] = &qword_4F8BA48[7];
  qword_4F8BA48[5] = 1;
  LODWORD(qword_4F8BA48[6]) = 0;
  BYTE4(qword_4F8BA48[6]) = 1;
  v3 = sub_C57470();
  v4 = LODWORD(qword_4F8BA48[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8BA48[1]) + 1 > HIDWORD(qword_4F8BA48[1]) )
  {
    v43 = v3;
    sub_C8D5F0(qword_4F8BA48, &qword_4F8BA48[2], LODWORD(qword_4F8BA48[1]) + 1LL, 8);
    v4 = LODWORD(qword_4F8BA48[1]);
    v3 = v43;
  }
  *(_QWORD *)(qword_4F8BA48[0] + 8 * v4) = v3;
  ++LODWORD(qword_4F8BA48[1]);
  qword_4F8BA48[8] = 0;
  qword_4F8BA48[9] = &unk_49D9748;
  qword_4F8BA48[10] = 0;
  qword_4F8BA00 = &unk_49DC090;
  qword_4F8BA48[11] = &unk_49DC1D0;
  qword_4F8BA48[15] = nullsub_23;
  qword_4F8BA48[14] = sub_984030;
  sub_C53080(&qword_4F8BA00, "reorder-sext-before-cnst-add", 28);
  LOBYTE(qword_4F8BA48[8]) = 0;
  LOWORD(qword_4F8BA48[10]) = 256;
  unk_4F8BA30 = 61;
  LOBYTE(word_4F8BA0C) = word_4F8BA0C & 0x98 | 0x21;
  unk_4F8BA28 = "Enable opt that reorders sext(add(a, CI)) to add(sext(a), CI)";
  sub_C53130(&qword_4F8BA00);
  __cxa_atexit(sub_984900, &qword_4F8BA00, &qword_4A427C0);
  v53 = "More aggresive floating point simplification";
  v54 = 44;
  v50 = 1;
  v51[0] = &v48;
  v49 = 1;
  v48 = 0;
  sub_F11640(&unk_4F8B920, "opt-use-fast-math", v51, &v49, &v50, &v53);
  __cxa_atexit(sub_984900, &unk_4F8B920, &qword_4A427C0);
  qword_4F8B840 = &unk_49DC150;
  v5 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8B84C = word_4F8B84C & 0x8000;
  unk_4F8B848 = v5;
  qword_4F8B888[1] = 0x100000000LL;
  unk_4F8B850 = 0;
  qword_4F8B888[0] = &qword_4F8B888[2];
  unk_4F8B858 = 0;
  unk_4F8B860 = 0;
  unk_4F8B868 = 0;
  unk_4F8B870 = 0;
  unk_4F8B878 = 0;
  unk_4F8B880 = 0;
  qword_4F8B888[3] = 0;
  qword_4F8B888[4] = &qword_4F8B888[7];
  qword_4F8B888[5] = 1;
  LODWORD(qword_4F8B888[6]) = 0;
  BYTE4(qword_4F8B888[6]) = 1;
  v6 = sub_C57470();
  v7 = LODWORD(qword_4F8B888[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8B888[1]) + 1 > HIDWORD(qword_4F8B888[1]) )
  {
    v44 = v6;
    sub_C8D5F0(qword_4F8B888, &qword_4F8B888[2], LODWORD(qword_4F8B888[1]) + 1LL, 8);
    v7 = LODWORD(qword_4F8B888[1]);
    v6 = v44;
  }
  *(_QWORD *)(qword_4F8B888[0] + 8 * v7) = v6;
  ++LODWORD(qword_4F8B888[1]);
  qword_4F8B888[8] = 0;
  qword_4F8B888[9] = &unk_49D9748;
  qword_4F8B888[10] = 0;
  qword_4F8B840 = &unk_49DC090;
  qword_4F8B888[11] = &unk_49DC1D0;
  qword_4F8B888[15] = nullsub_23;
  qword_4F8B888[14] = sub_984030;
  sub_C53080(&qword_4F8B840, "opt-use-prec-div", 16);
  LOBYTE(qword_4F8B888[8]) = 1;
  LOWORD(qword_4F8B888[10]) = 257;
  unk_4F8B870 = 28;
  LOBYTE(word_4F8B84C) = word_4F8B84C & 0x98 | 0x21;
  unk_4F8B868 = "Don't use fast approximation";
  sub_C53130(&qword_4F8B840);
  __cxa_atexit(sub_984900, &qword_4F8B840, &qword_4A427C0);
  qword_4F8B760 = &unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8B76C = word_4F8B76C & 0x8000;
  unk_4F8B770 = 0;
  qword_4F8B7A8[1] = 0x100000000LL;
  unk_4F8B768 = v8;
  qword_4F8B7A8[0] = &qword_4F8B7A8[2];
  unk_4F8B778 = 0;
  unk_4F8B780 = 0;
  unk_4F8B788 = 0;
  unk_4F8B790 = 0;
  unk_4F8B798 = 0;
  unk_4F8B7A0 = 0;
  qword_4F8B7A8[3] = 0;
  qword_4F8B7A8[4] = &qword_4F8B7A8[7];
  qword_4F8B7A8[5] = 1;
  LODWORD(qword_4F8B7A8[6]) = 0;
  BYTE4(qword_4F8B7A8[6]) = 1;
  v9 = sub_C57470();
  v10 = LODWORD(qword_4F8B7A8[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8B7A8[1]) + 1 > HIDWORD(qword_4F8B7A8[1]) )
  {
    v45 = v9;
    sub_C8D5F0(qword_4F8B7A8, &qword_4F8B7A8[2], LODWORD(qword_4F8B7A8[1]) + 1LL, 8);
    v10 = LODWORD(qword_4F8B7A8[1]);
    v9 = v45;
  }
  *(_QWORD *)(qword_4F8B7A8[0] + 8 * v10) = v9;
  ++LODWORD(qword_4F8B7A8[1]);
  qword_4F8B7A8[8] = 0;
  qword_4F8B7A8[9] = &unk_49D9748;
  qword_4F8B7A8[10] = 0;
  qword_4F8B760 = &unk_49DC090;
  qword_4F8B7A8[11] = &unk_49DC1D0;
  qword_4F8B7A8[15] = nullsub_23;
  qword_4F8B7A8[14] = sub_984030;
  sub_C53080(&qword_4F8B760, "disable-fp-cast-opt", 19);
  LOBYTE(qword_4F8B7A8[8]) = 0;
  LOWORD(qword_4F8B7A8[10]) = 256;
  unk_4F8B790 = 31;
  LOBYTE(word_4F8B76C) = word_4F8B76C & 0x98 | 0x21;
  unk_4F8B788 = "Disabling fp cast optimizations";
  sub_C53130(&qword_4F8B760);
  __cxa_atexit(sub_984900, &qword_4F8B760, &qword_4A427C0);
  v54 = 71;
  v53 = "Disable sinking, deprecated in favor of -instcombine-code-sinking=false";
  v50 = 1;
  LOBYTE(v49) = 0;
  v51[0] = &v49;
  sub_F11860(&unk_4F8B680, "disable-sink", v51, &v50, &v53);
  __cxa_atexit(sub_984900, &unk_4F8B680, &qword_4A427C0);
  v51[0] = &v49;
  v54 = 15;
  v53 = "Partial sinking";
  v50 = 1;
  LOBYTE(v49) = 1;
  sub_F11860(&unk_4F8B5A0, "partial-sink", v51, &v50, &v53);
  __cxa_atexit(sub_984900, &unk_4F8B5A0, &qword_4A427C0);
  qword_4F8B4C0 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F8B4CC &= 0x8000u;
  word_4F8B4D0 = 0;
  qword_4F8B510 = 0x100000000LL;
  qword_4F8B4D8 = 0;
  qword_4F8B4E0 = 0;
  qword_4F8B4E8 = 0;
  dword_4F8B4C8 = v11;
  qword_4F8B4F0 = 0;
  qword_4F8B4F8 = 0;
  qword_4F8B500 = 0;
  qword_4F8B508 = (__int64)&unk_4F8B518;
  qword_4F8B520 = 0;
  qword_4F8B528 = (__int64)&unk_4F8B540;
  qword_4F8B530 = 1;
  dword_4F8B538 = 0;
  byte_4F8B53C = 1;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_4F8B510;
  v14 = (unsigned int)qword_4F8B510 + 1LL;
  if ( v14 > HIDWORD(qword_4F8B510) )
  {
    sub_C8D5F0((char *)&unk_4F8B518 - 16, &unk_4F8B518, v14, 8);
    v13 = (unsigned int)qword_4F8B510;
  }
  *(_QWORD *)(qword_4F8B508 + 8 * v13) = v12;
  LODWORD(qword_4F8B510) = qword_4F8B510 + 1;
  qword_4F8B548 = 0;
  qword_4F8B550 = (__int64)&unk_49D9748;
  qword_4F8B558 = 0;
  qword_4F8B4C0 = (__int64)&unk_49DC090;
  qword_4F8B560 = (__int64)&unk_49DC1D0;
  qword_4F8B580 = (__int64)nullsub_23;
  qword_4F8B578 = (__int64)sub_984030;
  sub_C53080(&qword_4F8B4C0, "instcombine-split-gep-chain", 27);
  LOWORD(qword_4F8B558) = 257;
  LOBYTE(qword_4F8B548) = 1;
  qword_4F8B4F0 = 46;
  LOBYTE(dword_4F8B4CC) = dword_4F8B4CC & 0x9F | 0x20;
  qword_4F8B4E8 = (__int64)"Enable spliting GEP chians to independent GEPs";
  sub_C53130(&qword_4F8B4C0);
  __cxa_atexit(sub_984900, &qword_4F8B4C0, &qword_4A427C0);
  qword_4F8B420 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8B438 = 0;
  word_4F8B430 = 0;
  qword_4F8B440 = 0;
  qword_4F8B448 = 0;
  dword_4F8B42C = dword_4F8B42C & 0x8000 | 0x20;
  qword_4F8B470 = 0x100000000LL;
  dword_4F8B428 = v15;
  qword_4F8B450 = 0;
  qword_4F8B458 = 0;
  qword_4F8B460 = 0;
  qword_4F8B468 = (__int64)&unk_4F8B478;
  qword_4F8B480 = 0;
  qword_4F8B488 = (__int64)&unk_4F8B4A0;
  qword_4F8B490 = 1;
  dword_4F8B498 = 0;
  byte_4F8B49C = 1;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_4F8B470;
  v18 = (unsigned int)qword_4F8B470 + 1LL;
  if ( v18 > HIDWORD(qword_4F8B470) )
  {
    sub_C8D5F0((char *)&unk_4F8B478 - 16, &unk_4F8B478, v18, 8);
    v17 = (unsigned int)qword_4F8B470;
  }
  *(_QWORD *)(qword_4F8B468 + 8 * v17) = v16;
  LODWORD(qword_4F8B470) = qword_4F8B470 + 1;
  qword_4F8B4A8 = 0;
  qword_4F8B420 = (__int64)&unk_49DC380;
  sub_C53080(&qword_4F8B420, "split-gep-chain", 15);
  qword_4F8B450 = 38;
  qword_4F8B448 = (__int64)"Alias for -instcombine-split-gep-chain";
  if ( qword_4F8B4A8 )
  {
    v19 = sub_CEADF0();
    v57 = 1;
    v53 = "cl::alias must only have one cl::aliasopt(...) specified!";
    v56 = 3;
    sub_C53280(&qword_4F8B420, &v53, 0, 0, v19);
  }
  qword_4F8B4A8 = (__int64)&qword_4F8B4C0;
  sub_C53EE0(&qword_4F8B420);
  __cxa_atexit(sub_C4FC50, &qword_4F8B420, &qword_4A427C0);
  qword_4F8B340 = (__int64)&unk_49DC150;
  v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8B390 = 0x100000000LL;
  dword_4F8B34C &= 0x8000u;
  word_4F8B350 = 0;
  qword_4F8B358 = 0;
  qword_4F8B360 = 0;
  dword_4F8B348 = v20;
  qword_4F8B368 = 0;
  qword_4F8B370 = 0;
  qword_4F8B378 = 0;
  qword_4F8B380 = 0;
  qword_4F8B388 = (__int64)&unk_4F8B398;
  qword_4F8B3A0 = 0;
  qword_4F8B3A8 = (__int64)&unk_4F8B3C0;
  qword_4F8B3B0 = 1;
  dword_4F8B3B8 = 0;
  byte_4F8B3BC = 1;
  v21 = sub_C57470();
  v22 = (unsigned int)qword_4F8B390;
  v23 = (unsigned int)qword_4F8B390 + 1LL;
  if ( v23 > HIDWORD(qword_4F8B390) )
  {
    sub_C8D5F0((char *)&unk_4F8B398 - 16, &unk_4F8B398, v23, 8);
    v22 = (unsigned int)qword_4F8B390;
  }
  *(_QWORD *)(qword_4F8B388 + 8 * v22) = v21;
  LODWORD(qword_4F8B390) = qword_4F8B390 + 1;
  qword_4F8B3C8 = 0;
  qword_4F8B3D0 = (__int64)&unk_49D9748;
  qword_4F8B3D8 = 0;
  qword_4F8B340 = (__int64)&unk_49DC090;
  qword_4F8B3E0 = (__int64)&unk_49DC1D0;
  qword_4F8B400 = (__int64)nullsub_23;
  qword_4F8B3F8 = (__int64)sub_984030;
  sub_C53080(&qword_4F8B340, "instcombine-canonicalize-geps-i8", 32);
  LOWORD(qword_4F8B3D8) = 257;
  LOBYTE(qword_4F8B3C8) = 1;
  qword_4F8B370 = 45;
  LOBYTE(dword_4F8B34C) = dword_4F8B34C & 0x9F | 0x20;
  qword_4F8B368 = (__int64)"Enable canonicalize constant GEPs to i8 type.";
  sub_C53130(&qword_4F8B340);
  __cxa_atexit(sub_984900, &qword_4F8B340, &qword_4A427C0);
  qword_4F8B260 = (__int64)&unk_49DC150;
  v24 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8B2B0 = 0x100000000LL;
  dword_4F8B26C &= 0x8000u;
  word_4F8B270 = 0;
  qword_4F8B278 = 0;
  qword_4F8B280 = 0;
  dword_4F8B268 = v24;
  qword_4F8B288 = 0;
  qword_4F8B290 = 0;
  qword_4F8B298 = 0;
  qword_4F8B2A0 = 0;
  qword_4F8B2A8 = (__int64)&unk_4F8B2B8;
  qword_4F8B2C0 = 0;
  qword_4F8B2C8 = (__int64)&unk_4F8B2E0;
  qword_4F8B2D0 = 1;
  dword_4F8B2D8 = 0;
  byte_4F8B2DC = 1;
  v25 = sub_C57470();
  v26 = (unsigned int)qword_4F8B2B0;
  v27 = (unsigned int)qword_4F8B2B0 + 1LL;
  if ( v27 > HIDWORD(qword_4F8B2B0) )
  {
    sub_C8D5F0((char *)&unk_4F8B2B8 - 16, &unk_4F8B2B8, v27, 8);
    v26 = (unsigned int)qword_4F8B2B0;
  }
  *(_QWORD *)(qword_4F8B2A8 + 8 * v26) = v25;
  LODWORD(qword_4F8B2B0) = qword_4F8B2B0 + 1;
  qword_4F8B2E8 = 0;
  qword_4F8B2F0 = (__int64)&unk_49D9748;
  qword_4F8B2F8 = 0;
  qword_4F8B260 = (__int64)&unk_49DC090;
  qword_4F8B300 = (__int64)&unk_49DC1D0;
  qword_4F8B320 = (__int64)nullsub_23;
  qword_4F8B318 = (__int64)sub_984030;
  sub_C53080(&qword_4F8B260, "preserve-integer-fma-patterns", 29);
  LOBYTE(qword_4F8B2E8) = 1;
  LOWORD(qword_4F8B2F8) = 257;
  qword_4F8B290 = 69;
  LOBYTE(dword_4F8B26C) = dword_4F8B26C & 0x9F | 0x20;
  qword_4F8B288 = (__int64)"Disable some InstCombine transforms that disturb integer FMA patterns";
  sub_C53130(&qword_4F8B260);
  __cxa_atexit(sub_984900, &qword_4F8B260, &qword_4A427C0);
  qword_4F8B180 = (__int64)&unk_49DC150;
  v28 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8B1D0 = 0x100000000LL;
  dword_4F8B18C &= 0x8000u;
  word_4F8B190 = 0;
  qword_4F8B198 = 0;
  qword_4F8B1A0 = 0;
  dword_4F8B188 = v28;
  qword_4F8B1A8 = 0;
  qword_4F8B1B0 = 0;
  qword_4F8B1B8 = 0;
  qword_4F8B1C0 = 0;
  qword_4F8B1C8 = (__int64)&unk_4F8B1D8;
  qword_4F8B1E0 = 0;
  qword_4F8B1E8 = (__int64)&unk_4F8B200;
  qword_4F8B1F0 = 1;
  dword_4F8B1F8 = 0;
  byte_4F8B1FC = 1;
  v29 = sub_C57470();
  v30 = (unsigned int)qword_4F8B1D0;
  v31 = (unsigned int)qword_4F8B1D0 + 1LL;
  if ( v31 > HIDWORD(qword_4F8B1D0) )
  {
    sub_C8D5F0((char *)&unk_4F8B1D8 - 16, &unk_4F8B1D8, v31, 8);
    v30 = (unsigned int)qword_4F8B1D0;
  }
  *(_QWORD *)(qword_4F8B1C8 + 8 * v30) = v29;
  LODWORD(qword_4F8B1D0) = qword_4F8B1D0 + 1;
  qword_4F8B208 = 0;
  qword_4F8B210 = (__int64)&unk_49D9748;
  qword_4F8B218 = 0;
  qword_4F8B180 = (__int64)&unk_49DC090;
  qword_4F8B220 = (__int64)&unk_49DC1D0;
  qword_4F8B240 = (__int64)nullsub_23;
  qword_4F8B238 = (__int64)sub_984030;
  sub_C53080(&qword_4F8B180, "instcombine-code-sinking", 24);
  LOWORD(qword_4F8B218) = 257;
  qword_4F8B1A8 = (__int64)"Enable code sinking";
  qword_4F8B1B0 = 19;
  LOBYTE(qword_4F8B208) = 1;
  sub_C53130(&qword_4F8B180);
  __cxa_atexit(sub_984900, &qword_4F8B180, &qword_4A427C0);
  qword_4F8B0A0 = (__int64)&unk_49DC150;
  v32 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8B0F0 = 0x100000000LL;
  dword_4F8B0AC &= 0x8000u;
  word_4F8B0B0 = 0;
  qword_4F8B0B8 = 0;
  qword_4F8B0C0 = 0;
  dword_4F8B0A8 = v32;
  qword_4F8B0C8 = 0;
  qword_4F8B0D0 = 0;
  qword_4F8B0D8 = 0;
  qword_4F8B0E0 = 0;
  qword_4F8B0E8 = (__int64)&unk_4F8B0F8;
  qword_4F8B100 = 0;
  qword_4F8B108 = (__int64)&unk_4F8B120;
  qword_4F8B110 = 1;
  dword_4F8B118 = 0;
  byte_4F8B11C = 1;
  v33 = sub_C57470();
  v34 = (unsigned int)qword_4F8B0F0;
  v35 = (unsigned int)qword_4F8B0F0 + 1LL;
  if ( v35 > HIDWORD(qword_4F8B0F0) )
  {
    sub_C8D5F0((char *)&unk_4F8B0F8 - 16, &unk_4F8B0F8, v35, 8);
    v34 = (unsigned int)qword_4F8B0F0;
  }
  *(_QWORD *)(qword_4F8B0E8 + 8 * v34) = v33;
  LODWORD(qword_4F8B0F0) = qword_4F8B0F0 + 1;
  qword_4F8B128 = 0;
  qword_4F8B130 = (__int64)&unk_49D9728;
  qword_4F8B0A0 = (__int64)&unk_49DBF10;
  qword_4F8B140 = (__int64)&unk_49DC290;
  qword_4F8B160 = (__int64)nullsub_24;
  qword_4F8B158 = (__int64)sub_984050;
  qword_4F8B138 = 0;
  sub_C53080(&qword_4F8B0A0, "instcombine-max-sink-users", 26);
  LODWORD(qword_4F8B128) = 32;
  qword_4F8B0C8 = (__int64)"Maximum number of undroppable users for instruction sinking";
  BYTE4(qword_4F8B138) = 1;
  LODWORD(qword_4F8B138) = 32;
  qword_4F8B0D0 = 59;
  sub_C53130(&qword_4F8B0A0);
  __cxa_atexit(sub_984970, &qword_4F8B0A0, &qword_4A427C0);
  qword_4F8AFC0 = (__int64)&unk_49DC150;
  v36 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F8B03C = 1;
  qword_4F8B010 = 0x100000000LL;
  dword_4F8AFCC &= 0x8000u;
  qword_4F8B008 = (__int64)&unk_4F8B018;
  qword_4F8AFD8 = 0;
  qword_4F8AFE0 = 0;
  dword_4F8AFC8 = v36;
  word_4F8AFD0 = 0;
  qword_4F8AFE8 = 0;
  qword_4F8AFF0 = 0;
  qword_4F8AFF8 = 0;
  qword_4F8B000 = 0;
  qword_4F8B020 = 0;
  qword_4F8B028 = (__int64)&unk_4F8B040;
  qword_4F8B030 = 1;
  dword_4F8B038 = 0;
  v37 = sub_C57470();
  v38 = (unsigned int)qword_4F8B010;
  if ( (unsigned __int64)(unsigned int)qword_4F8B010 + 1 > HIDWORD(qword_4F8B010) )
  {
    v46 = v37;
    sub_C8D5F0((char *)&unk_4F8B018 - 16, &unk_4F8B018, (unsigned int)qword_4F8B010 + 1LL, 8);
    v38 = (unsigned int)qword_4F8B010;
    v37 = v46;
  }
  *(_QWORD *)(qword_4F8B008 + 8 * v38) = v37;
  LODWORD(qword_4F8B010) = qword_4F8B010 + 1;
  qword_4F8B080 = (__int64)nullsub_24;
  qword_4F8B050 = (__int64)&unk_49D9728;
  qword_4F8AFC0 = (__int64)&unk_49DBF10;
  qword_4F8B060 = (__int64)&unk_49DC290;
  qword_4F8B048 = 0;
  qword_4F8B078 = (__int64)sub_984050;
  qword_4F8B058 = 0;
  sub_C53080(&qword_4F8AFC0, "instcombine-maxarray-size", 25);
  LODWORD(qword_4F8B048) = 1024;
  qword_4F8AFE8 = (__int64)"Maximum array size considered when doing a combine";
  BYTE4(qword_4F8B058) = 1;
  LODWORD(qword_4F8B058) = 1024;
  qword_4F8AFF0 = 50;
  sub_C53130(&qword_4F8AFC0);
  __cxa_atexit(sub_984970, &qword_4F8AFC0, &qword_4A427C0);
  qword_4F8AEE0 = (__int64)&unk_49DC150;
  v39 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F8AEEC &= 0x8000u;
  word_4F8AEF0 = 0;
  qword_4F8AF30 = 0x100000000LL;
  qword_4F8AF28 = (__int64)&unk_4F8AF38;
  qword_4F8AEF8 = 0;
  qword_4F8AF00 = 0;
  dword_4F8AEE8 = v39;
  qword_4F8AF08 = 0;
  qword_4F8AF10 = 0;
  qword_4F8AF18 = 0;
  qword_4F8AF20 = 0;
  qword_4F8AF40 = 0;
  qword_4F8AF48 = (__int64)&unk_4F8AF60;
  qword_4F8AF50 = 1;
  dword_4F8AF58 = 0;
  byte_4F8AF5C = 1;
  v40 = sub_C57470();
  v41 = (unsigned int)qword_4F8AF30;
  if ( (unsigned __int64)(unsigned int)qword_4F8AF30 + 1 > HIDWORD(qword_4F8AF30) )
  {
    v47 = v40;
    sub_C8D5F0((char *)&unk_4F8AF38 - 16, &unk_4F8AF38, (unsigned int)qword_4F8AF30 + 1LL, 8);
    v41 = (unsigned int)qword_4F8AF30;
    v40 = v47;
  }
  *(_QWORD *)(qword_4F8AF28 + 8 * v41) = v40;
  qword_4F8AFA0 = (__int64)nullsub_24;
  LODWORD(qword_4F8AF30) = qword_4F8AF30 + 1;
  qword_4F8AF70 = (__int64)&unk_49D9728;
  qword_4F8AEE0 = (__int64)&unk_49DBF10;
  qword_4F8AF80 = (__int64)&unk_49DC290;
  qword_4F8AF68 = 0;
  qword_4F8AF98 = (__int64)sub_984050;
  qword_4F8AF78 = 0;
  sub_C53080(&qword_4F8AEE0, "instcombine-lower-dbg-declare", 29);
  LODWORD(qword_4F8AF68) = 1;
  BYTE4(qword_4F8AF78) = 1;
  LODWORD(qword_4F8AF78) = 1;
  LOBYTE(dword_4F8AEEC) = dword_4F8AEEC & 0x9F | 0x20;
  sub_C53130(&qword_4F8AEE0);
  return __cxa_atexit(sub_984970, &qword_4F8AEE0, &qword_4A427C0);
}

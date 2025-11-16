// Function: ctor_650_0
// Address: 0x598640
//
int ctor_650_0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  _QWORD *v2; // rsi
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // rdi
  int v6; // edx
  __int64 v7; // r12
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rcx
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rcx
  int v17; // edx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rcx
  int v22; // edx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rcx
  int v27; // edx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rcx
  int v32; // edx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rcx
  int v37; // edx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rcx
  int v42; // edx
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rcx
  int v47; // edx
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rdx
  __int64 v51; // rcx
  int v52; // edx
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // rcx
  int v57; // edx
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rdx
  __int64 v61; // rcx
  int v62; // edx
  __int64 v63; // rbx
  __int64 v64; // rax
  unsigned __int64 v65; // rdx
  __int64 v67; // [rsp+8h] [rbp-78h]
  __int64 v68; // [rsp+8h] [rbp-78h]
  __int64 v69; // [rsp+8h] [rbp-78h]
  __int64 v70; // [rsp+8h] [rbp-78h]
  __int64 v71; // [rsp+8h] [rbp-78h]
  __int64 v72; // [rsp+8h] [rbp-78h]
  __int64 v73; // [rsp+8h] [rbp-78h]
  __int64 v74; // [rsp+8h] [rbp-78h]
  __int64 v75; // [rsp+8h] [rbp-78h]
  __int64 v76; // [rsp+8h] [rbp-78h]
  _QWORD v77[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v78[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v79[2]; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v80[8]; // [rsp+40h] [rbp-40h] BYREF

  v0 = sub_C60B10();
  v79[0] = v80;
  v1 = v0;
  sub_325EC10(v79, "Controls whether a DAG combine is performed for a node");
  v77[0] = v78;
  sub_325EC10(v77, "dagcombine");
  v2 = v77;
  sub_CF9810(v1, v77, v79);
  if ( (_QWORD *)v77[0] != v78 )
  {
    v2 = (_QWORD *)(v78[0] + 1LL);
    j_j___libc_free_0(v77[0], v78[0] + 1LL);
  }
  v5 = v79[0];
  if ( (_QWORD *)v79[0] != v80 )
  {
    v2 = (_QWORD *)(v80[0] + 1LL);
    j_j___libc_free_0(v79[0], v80[0] + 1LL);
  }
  qword_5038460 = (__int64)&unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(v5, v2, v3, v4), 1u);
  qword_50384B0 = 0x100000000LL;
  dword_503846C &= 0x8000u;
  word_5038470 = 0;
  qword_5038478 = 0;
  qword_5038480 = 0;
  dword_5038468 = v6;
  qword_5038488 = 0;
  qword_5038490 = 0;
  qword_5038498 = 0;
  qword_50384A0 = 0;
  qword_50384A8 = (__int64)&unk_50384B8;
  qword_50384C0 = 0;
  qword_50384C8 = (__int64)&unk_50384E0;
  qword_50384D0 = 1;
  dword_50384D8 = 0;
  byte_50384DC = 1;
  v7 = sub_C57470();
  v8 = (unsigned int)qword_50384B0;
  v9 = (unsigned int)qword_50384B0 + 1LL;
  if ( v9 > HIDWORD(qword_50384B0) )
  {
    sub_C8D5F0((char *)&unk_50384B8 - 16, &unk_50384B8, v9, 8);
    v8 = (unsigned int)qword_50384B0;
  }
  *(_QWORD *)(qword_50384A8 + 8 * v8) = v7;
  qword_50384F0 = (__int64)&unk_49D9748;
  qword_5038460 = (__int64)&unk_49DC090;
  qword_5038500 = (__int64)&unk_49DC1D0;
  LODWORD(qword_50384B0) = qword_50384B0 + 1;
  qword_5038520 = (__int64)nullsub_23;
  qword_50384E8 = 0;
  qword_5038518 = (__int64)sub_984030;
  qword_50384F8 = 0;
  sub_C53080(&qword_5038460, "combiner-global-alias-analysis", 30);
  qword_5038490 = 46;
  LOBYTE(dword_503846C) = dword_503846C & 0x9F | 0x20;
  qword_5038488 = (__int64)"Enable DAG combiner's use of IR alias analysis";
  sub_C53130(&qword_5038460);
  __cxa_atexit(sub_984900, &qword_5038460, &qword_4A427C0);
  qword_5038380 = (__int64)&unk_49DC150;
  v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5038460, v10, v11), 1u);
  qword_50383D0 = 0x100000000LL;
  dword_503838C &= 0x8000u;
  qword_50383C8 = (__int64)&unk_50383D8;
  word_5038390 = 0;
  qword_5038398 = 0;
  dword_5038388 = v12;
  qword_50383A0 = 0;
  qword_50383A8 = 0;
  qword_50383B0 = 0;
  qword_50383B8 = 0;
  qword_50383C0 = 0;
  qword_50383E0 = 0;
  qword_50383E8 = (__int64)&unk_5038400;
  qword_50383F0 = 1;
  dword_50383F8 = 0;
  byte_50383FC = 1;
  v13 = sub_C57470();
  v14 = (unsigned int)qword_50383D0;
  if ( (unsigned __int64)(unsigned int)qword_50383D0 + 1 > HIDWORD(qword_50383D0) )
  {
    v67 = v13;
    sub_C8D5F0((char *)&unk_50383D8 - 16, &unk_50383D8, (unsigned int)qword_50383D0 + 1LL, 8);
    v14 = (unsigned int)qword_50383D0;
    v13 = v67;
  }
  *(_QWORD *)(qword_50383C8 + 8 * v14) = v13;
  qword_5038410 = (__int64)&unk_49D9748;
  qword_5038380 = (__int64)&unk_49DC090;
  qword_5038420 = (__int64)&unk_49DC1D0;
  LODWORD(qword_50383D0) = qword_50383D0 + 1;
  qword_5038440 = (__int64)nullsub_23;
  qword_5038408 = 0;
  qword_5038438 = (__int64)sub_984030;
  qword_5038418 = 0;
  sub_C53080(&qword_5038380, "combiner-use-tbaa", 17);
  LOBYTE(qword_5038408) = 1;
  LOWORD(qword_5038418) = 257;
  qword_50383B0 = 33;
  LOBYTE(dword_503838C) = dword_503838C & 0x9F | 0x20;
  qword_50383A8 = (__int64)"Enable DAG combiner's use of TBAA";
  sub_C53130(&qword_5038380);
  __cxa_atexit(sub_984900, &qword_5038380, &qword_4A427C0);
  qword_50382A0 = (__int64)&unk_49DC150;
  v17 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5038380, v15, v16), 1u);
  qword_50382F0 = 0x100000000LL;
  dword_50382AC &= 0x8000u;
  word_50382B0 = 0;
  qword_50382E8 = (__int64)&unk_50382F8;
  qword_50382B8 = 0;
  dword_50382A8 = v17;
  qword_50382C0 = 0;
  qword_50382C8 = 0;
  qword_50382D0 = 0;
  qword_50382D8 = 0;
  qword_50382E0 = 0;
  qword_5038300 = 0;
  qword_5038308 = (__int64)&unk_5038320;
  qword_5038310 = 1;
  dword_5038318 = 0;
  byte_503831C = 1;
  v18 = sub_C57470();
  v19 = (unsigned int)qword_50382F0;
  if ( (unsigned __int64)(unsigned int)qword_50382F0 + 1 > HIDWORD(qword_50382F0) )
  {
    v68 = v18;
    sub_C8D5F0((char *)&unk_50382F8 - 16, &unk_50382F8, (unsigned int)qword_50382F0 + 1LL, 8);
    v19 = (unsigned int)qword_50382F0;
    v18 = v68;
  }
  *(_QWORD *)(qword_50382E8 + 8 * v19) = v18;
  qword_5038330 = (__int64)&unk_49D9748;
  qword_50382A0 = (__int64)&unk_49DC090;
  qword_5038340 = (__int64)&unk_49DC1D0;
  LODWORD(qword_50382F0) = qword_50382F0 + 1;
  qword_5038360 = (__int64)nullsub_23;
  qword_5038328 = 0;
  qword_5038358 = (__int64)sub_984030;
  qword_5038338 = 0;
  sub_C53080(&qword_50382A0, "combiner-stress-load-slicing", 28);
  LOWORD(qword_5038338) = 256;
  LOBYTE(qword_5038328) = 0;
  qword_50382D0 = 46;
  LOBYTE(dword_50382AC) = dword_50382AC & 0x9F | 0x20;
  qword_50382C8 = (__int64)"Bypass the profitability model of load slicing";
  sub_C53130(&qword_50382A0);
  __cxa_atexit(sub_984900, &qword_50382A0, &qword_4A427C0);
  qword_50381C0 = (__int64)&unk_49DC150;
  v22 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_50382A0, v20, v21), 1u);
  byte_503823C = 1;
  word_50381D0 = 0;
  qword_5038210 = 0x100000000LL;
  dword_50381CC &= 0x8000u;
  qword_5038208 = (__int64)&unk_5038218;
  qword_50381D8 = 0;
  dword_50381C8 = v22;
  qword_50381E0 = 0;
  qword_50381E8 = 0;
  qword_50381F0 = 0;
  qword_50381F8 = 0;
  qword_5038200 = 0;
  qword_5038220 = 0;
  qword_5038228 = (__int64)&unk_5038240;
  qword_5038230 = 1;
  dword_5038238 = 0;
  v23 = sub_C57470();
  v24 = (unsigned int)qword_5038210;
  if ( (unsigned __int64)(unsigned int)qword_5038210 + 1 > HIDWORD(qword_5038210) )
  {
    v69 = v23;
    sub_C8D5F0((char *)&unk_5038218 - 16, &unk_5038218, (unsigned int)qword_5038210 + 1LL, 8);
    v24 = (unsigned int)qword_5038210;
    v23 = v69;
  }
  *(_QWORD *)(qword_5038208 + 8 * v24) = v23;
  qword_5038250 = (__int64)&unk_49D9748;
  qword_50381C0 = (__int64)&unk_49DC090;
  qword_5038260 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5038210) = qword_5038210 + 1;
  qword_5038280 = (__int64)nullsub_23;
  qword_5038248 = 0;
  qword_5038278 = (__int64)sub_984030;
  qword_5038258 = 0;
  sub_C53080(&qword_50381C0, "combiner-split-load-index", 25);
  LOBYTE(qword_5038248) = 1;
  qword_50381F0 = 42;
  LOBYTE(dword_50381CC) = dword_50381CC & 0x9F | 0x20;
  LOWORD(qword_5038258) = 257;
  qword_50381E8 = (__int64)"DAG combiner may split indexing from loads";
  sub_C53130(&qword_50381C0);
  __cxa_atexit(sub_984900, &qword_50381C0, &qword_4A427C0);
  qword_50380E0 = (__int64)&unk_49DC150;
  v27 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_50381C0, v25, v26), 1u);
  byte_503815C = 1;
  qword_5038130 = 0x100000000LL;
  dword_50380EC &= 0x8000u;
  qword_5038128 = (__int64)&unk_5038138;
  qword_50380F8 = 0;
  qword_5038100 = 0;
  dword_50380E8 = v27;
  word_50380F0 = 0;
  qword_5038108 = 0;
  qword_5038110 = 0;
  qword_5038118 = 0;
  qword_5038120 = 0;
  qword_5038140 = 0;
  qword_5038148 = (__int64)&unk_5038160;
  qword_5038150 = 1;
  dword_5038158 = 0;
  v28 = sub_C57470();
  v29 = (unsigned int)qword_5038130;
  if ( (unsigned __int64)(unsigned int)qword_5038130 + 1 > HIDWORD(qword_5038130) )
  {
    v70 = v28;
    sub_C8D5F0((char *)&unk_5038138 - 16, &unk_5038138, (unsigned int)qword_5038130 + 1LL, 8);
    v29 = (unsigned int)qword_5038130;
    v28 = v70;
  }
  *(_QWORD *)(qword_5038128 + 8 * v29) = v28;
  qword_5038170 = (__int64)&unk_49D9748;
  qword_50380E0 = (__int64)&unk_49DC090;
  qword_5038180 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5038130) = qword_5038130 + 1;
  qword_50381A0 = (__int64)nullsub_23;
  qword_5038168 = 0;
  qword_5038198 = (__int64)sub_984030;
  qword_5038178 = 0;
  sub_C53080(&qword_50380E0, "combiner-store-merging", 22);
  LOBYTE(qword_5038168) = 1;
  qword_5038110 = 62;
  LOBYTE(dword_50380EC) = dword_50380EC & 0x9F | 0x20;
  LOWORD(qword_5038178) = 257;
  qword_5038108 = (__int64)"DAG combiner enable merging multiple stores into a wider store";
  sub_C53130(&qword_50380E0);
  __cxa_atexit(sub_984900, &qword_50380E0, &qword_4A427C0);
  qword_5038000 = (__int64)&unk_49DC150;
  v32 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_50380E0, v30, v31), 1u);
  dword_503800C &= 0x8000u;
  word_5038010 = 0;
  qword_5038050 = 0x100000000LL;
  qword_5038048 = (__int64)&unk_5038058;
  qword_5038018 = 0;
  qword_5038020 = 0;
  dword_5038008 = v32;
  qword_5038028 = 0;
  qword_5038030 = 0;
  qword_5038038 = 0;
  qword_5038040 = 0;
  qword_5038060 = 0;
  qword_5038068 = (__int64)&unk_5038080;
  qword_5038070 = 1;
  dword_5038078 = 0;
  byte_503807C = 1;
  v33 = sub_C57470();
  v34 = (unsigned int)qword_5038050;
  if ( (unsigned __int64)(unsigned int)qword_5038050 + 1 > HIDWORD(qword_5038050) )
  {
    v71 = v33;
    sub_C8D5F0((char *)&unk_5038058 - 16, &unk_5038058, (unsigned int)qword_5038050 + 1LL, 8);
    v34 = (unsigned int)qword_5038050;
    v33 = v71;
  }
  *(_QWORD *)(qword_5038048 + 8 * v34) = v33;
  LODWORD(qword_5038050) = qword_5038050 + 1;
  qword_5038088 = 0;
  qword_5038090 = (__int64)&unk_49D9728;
  qword_5038098 = 0;
  qword_5038000 = (__int64)&unk_49DBF10;
  qword_50380A0 = (__int64)&unk_49DC290;
  qword_50380C0 = (__int64)nullsub_24;
  qword_50380B8 = (__int64)sub_984050;
  sub_C53080(&qword_5038000, "combiner-tokenfactor-inline-limit", 33);
  LODWORD(qword_5038088) = 2048;
  BYTE4(qword_5038098) = 1;
  LODWORD(qword_5038098) = 2048;
  qword_5038030 = 56;
  LOBYTE(dword_503800C) = dword_503800C & 0x9F | 0x20;
  qword_5038028 = (__int64)"Limit the number of operands to inline for Token Factors";
  sub_C53130(&qword_5038000);
  __cxa_atexit(sub_984970, &qword_5038000, &qword_4A427C0);
  qword_5037F20 = (__int64)&unk_49DC150;
  v37 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_5038000, v35, v36), 1u);
  dword_5037F2C &= 0x8000u;
  word_5037F30 = 0;
  qword_5037F70 = 0x100000000LL;
  qword_5037F68 = (__int64)&unk_5037F78;
  qword_5037F38 = 0;
  qword_5037F40 = 0;
  dword_5037F28 = v37;
  qword_5037F48 = 0;
  qword_5037F50 = 0;
  qword_5037F58 = 0;
  qword_5037F60 = 0;
  qword_5037F80 = 0;
  qword_5037F88 = (__int64)&unk_5037FA0;
  qword_5037F90 = 1;
  dword_5037F98 = 0;
  byte_5037F9C = 1;
  v38 = sub_C57470();
  v39 = (unsigned int)qword_5037F70;
  if ( (unsigned __int64)(unsigned int)qword_5037F70 + 1 > HIDWORD(qword_5037F70) )
  {
    v72 = v38;
    sub_C8D5F0((char *)&unk_5037F78 - 16, &unk_5037F78, (unsigned int)qword_5037F70 + 1LL, 8);
    v39 = (unsigned int)qword_5037F70;
    v38 = v72;
  }
  *(_QWORD *)(qword_5037F68 + 8 * v39) = v38;
  LODWORD(qword_5037F70) = qword_5037F70 + 1;
  qword_5037FA8 = 0;
  qword_5037FB0 = (__int64)&unk_49D9728;
  qword_5037FB8 = 0;
  qword_5037F20 = (__int64)&unk_49DBF10;
  qword_5037FC0 = (__int64)&unk_49DC290;
  qword_5037FE0 = (__int64)nullsub_24;
  qword_5037FD8 = (__int64)sub_984050;
  sub_C53080(&qword_5037F20, "combiner-store-merge-dependence-limit", 37);
  LODWORD(qword_5037FA8) = 10;
  BYTE4(qword_5037FB8) = 1;
  LODWORD(qword_5037FB8) = 10;
  qword_5037F50 = 107;
  LOBYTE(dword_5037F2C) = dword_5037F2C & 0x9F | 0x20;
  qword_5037F48 = (__int64)"Limit the number of times for the same StoreNode and RootNode to bail out in store merging dependence check";
  sub_C53130(&qword_5037F20);
  __cxa_atexit(sub_984970, &qword_5037F20, &qword_4A427C0);
  qword_5037E40 = (__int64)&unk_49DC150;
  v42 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_5037F20, v40, v41), 1u);
  dword_5037E4C &= 0x8000u;
  word_5037E50 = 0;
  qword_5037E90 = 0x100000000LL;
  qword_5037E88 = (__int64)&unk_5037E98;
  qword_5037E58 = 0;
  qword_5037E60 = 0;
  dword_5037E48 = v42;
  qword_5037E68 = 0;
  qword_5037E70 = 0;
  qword_5037E78 = 0;
  qword_5037E80 = 0;
  qword_5037EA0 = 0;
  qword_5037EA8 = (__int64)&unk_5037EC0;
  qword_5037EB0 = 1;
  dword_5037EB8 = 0;
  byte_5037EBC = 1;
  v43 = sub_C57470();
  v44 = (unsigned int)qword_5037E90;
  if ( (unsigned __int64)(unsigned int)qword_5037E90 + 1 > HIDWORD(qword_5037E90) )
  {
    v73 = v43;
    sub_C8D5F0((char *)&unk_5037E98 - 16, &unk_5037E98, (unsigned int)qword_5037E90 + 1LL, 8);
    v44 = (unsigned int)qword_5037E90;
    v43 = v73;
  }
  *(_QWORD *)(qword_5037E88 + 8 * v44) = v43;
  qword_5037ED0 = (__int64)&unk_49D9748;
  qword_5037E40 = (__int64)&unk_49DC090;
  qword_5037EE0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5037E90) = qword_5037E90 + 1;
  qword_5037F00 = (__int64)nullsub_23;
  qword_5037EC8 = 0;
  qword_5037EF8 = (__int64)sub_984030;
  qword_5037ED8 = 0;
  sub_C53080(&qword_5037E40, "combiner-reduce-load-op-store-width", 35);
  LOWORD(qword_5037ED8) = 257;
  LOBYTE(qword_5037EC8) = 1;
  qword_5037E70 = 64;
  LOBYTE(dword_5037E4C) = dword_5037E4C & 0x9F | 0x20;
  qword_5037E68 = (__int64)"DAG combiner enable reducing the width of load/op/store sequence";
  sub_C53130(&qword_5037E40);
  __cxa_atexit(sub_984900, &qword_5037E40, &qword_4A427C0);
  qword_5037D60 = (__int64)&unk_49DC150;
  v47 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5037E40, v45, v46), 1u);
  qword_5037DB0 = 0x100000000LL;
  dword_5037D6C &= 0x8000u;
  qword_5037DA8 = (__int64)&unk_5037DB8;
  word_5037D70 = 0;
  qword_5037D78 = 0;
  dword_5037D68 = v47;
  qword_5037D80 = 0;
  qword_5037D88 = 0;
  qword_5037D90 = 0;
  qword_5037D98 = 0;
  qword_5037DA0 = 0;
  qword_5037DC0 = 0;
  qword_5037DC8 = (__int64)&unk_5037DE0;
  qword_5037DD0 = 1;
  dword_5037DD8 = 0;
  byte_5037DDC = 1;
  v48 = sub_C57470();
  v49 = (unsigned int)qword_5037DB0;
  if ( (unsigned __int64)(unsigned int)qword_5037DB0 + 1 > HIDWORD(qword_5037DB0) )
  {
    v74 = v48;
    sub_C8D5F0((char *)&unk_5037DB8 - 16, &unk_5037DB8, (unsigned int)qword_5037DB0 + 1LL, 8);
    v49 = (unsigned int)qword_5037DB0;
    v48 = v74;
  }
  *(_QWORD *)(qword_5037DA8 + 8 * v49) = v48;
  qword_5037DF0 = (__int64)&unk_49D9748;
  qword_5037D60 = (__int64)&unk_49DC090;
  qword_5037E00 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5037DB0) = qword_5037DB0 + 1;
  qword_5037E20 = (__int64)nullsub_23;
  qword_5037DE8 = 0;
  qword_5037E18 = (__int64)sub_984030;
  qword_5037DF8 = 0;
  sub_C53080(&qword_5037D60, "combiner-reduce-load-op-store-width-force-narrowing-profitable", 62);
  LOWORD(qword_5037DF8) = 256;
  LOBYTE(qword_5037DE8) = 0;
  qword_5037D90 = 109;
  LOBYTE(dword_5037D6C) = dword_5037D6C & 0x9F | 0x20;
  qword_5037D88 = (__int64)"DAG combiner force override the narrowing profitable check when reducing the width of load/op"
                           "/store sequences";
  sub_C53130(&qword_5037D60);
  __cxa_atexit(sub_984900, &qword_5037D60, &qword_4A427C0);
  qword_5037C80 = (__int64)&unk_49DC150;
  v52 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5037D60, v50, v51), 1u);
  qword_5037CD0 = 0x100000000LL;
  dword_5037C8C &= 0x8000u;
  qword_5037CC8 = (__int64)&unk_5037CD8;
  word_5037C90 = 0;
  qword_5037C98 = 0;
  dword_5037C88 = v52;
  qword_5037CA0 = 0;
  qword_5037CA8 = 0;
  qword_5037CB0 = 0;
  qword_5037CB8 = 0;
  qword_5037CC0 = 0;
  qword_5037CE0 = 0;
  qword_5037CE8 = (__int64)&unk_5037D00;
  qword_5037CF0 = 1;
  dword_5037CF8 = 0;
  byte_5037CFC = 1;
  v53 = sub_C57470();
  v54 = (unsigned int)qword_5037CD0;
  if ( (unsigned __int64)(unsigned int)qword_5037CD0 + 1 > HIDWORD(qword_5037CD0) )
  {
    v75 = v53;
    sub_C8D5F0((char *)&unk_5037CD8 - 16, &unk_5037CD8, (unsigned int)qword_5037CD0 + 1LL, 8);
    v54 = (unsigned int)qword_5037CD0;
    v53 = v75;
  }
  *(_QWORD *)(qword_5037CC8 + 8 * v54) = v53;
  qword_5037D10 = (__int64)&unk_49D9748;
  qword_5037C80 = (__int64)&unk_49DC090;
  qword_5037D20 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5037CD0) = qword_5037CD0 + 1;
  qword_5037D40 = (__int64)nullsub_23;
  qword_5037D08 = 0;
  qword_5037D38 = (__int64)sub_984030;
  qword_5037D18 = 0;
  sub_C53080(&qword_5037C80, "combiner-shrink-load-replace-store-with-store", 45);
  LOWORD(qword_5037D18) = 257;
  LOBYTE(qword_5037D08) = 1;
  qword_5037CB0 = 68;
  LOBYTE(dword_5037C8C) = dword_5037C8C & 0x9F | 0x20;
  qword_5037CA8 = (__int64)"DAG combiner enable load/<replace bytes>/store with a narrower store";
  sub_C53130(&qword_5037C80);
  __cxa_atexit(sub_984900, &qword_5037C80, &qword_4A427C0);
  qword_5037BA0 = (__int64)&unk_49DC150;
  v57 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5037C80, v55, v56), 1u);
  qword_5037BF0 = 0x100000000LL;
  dword_5037BAC &= 0x8000u;
  qword_5037BE8 = (__int64)&unk_5037BF8;
  word_5037BB0 = 0;
  qword_5037BB8 = 0;
  dword_5037BA8 = v57;
  qword_5037BC0 = 0;
  qword_5037BC8 = 0;
  qword_5037BD0 = 0;
  qword_5037BD8 = 0;
  qword_5037BE0 = 0;
  qword_5037C00 = 0;
  qword_5037C08 = (__int64)&unk_5037C20;
  qword_5037C10 = 1;
  dword_5037C18 = 0;
  byte_5037C1C = 1;
  v58 = sub_C57470();
  v59 = (unsigned int)qword_5037BF0;
  if ( (unsigned __int64)(unsigned int)qword_5037BF0 + 1 > HIDWORD(qword_5037BF0) )
  {
    v76 = v58;
    sub_C8D5F0((char *)&unk_5037BF8 - 16, &unk_5037BF8, (unsigned int)qword_5037BF0 + 1LL, 8);
    v59 = (unsigned int)qword_5037BF0;
    v58 = v76;
  }
  *(_QWORD *)(qword_5037BE8 + 8 * v59) = v58;
  qword_5037C30 = (__int64)&unk_49D9748;
  qword_5037BA0 = (__int64)&unk_49DC090;
  qword_5037C40 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5037BF0) = qword_5037BF0 + 1;
  qword_5037C60 = (__int64)nullsub_23;
  qword_5037C28 = 0;
  qword_5037C58 = (__int64)sub_984030;
  qword_5037C38 = 0;
  sub_C53080(&qword_5037BA0, "combiner-add-rotate", 19);
  LOWORD(qword_5037C38) = 257;
  LOBYTE(qword_5037C28) = 1;
  qword_5037BD0 = 41;
  LOBYTE(dword_5037BAC) = dword_5037BAC & 0x9F | 0x20;
  qword_5037BC8 = (__int64)"DAG combiner matches rotation on addition";
  sub_C53130(&qword_5037BA0);
  __cxa_atexit(sub_984900, &qword_5037BA0, &qword_4A427C0);
  qword_5037AC0 = (__int64)&unk_49DC150;
  v62 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5037BA0, v60, v61), 1u);
  qword_5037B10 = 0x100000000LL;
  dword_5037ACC &= 0x8000u;
  word_5037AD0 = 0;
  qword_5037B08 = (__int64)&unk_5037B18;
  qword_5037AD8 = 0;
  dword_5037AC8 = v62;
  qword_5037AE0 = 0;
  qword_5037AE8 = 0;
  qword_5037AF0 = 0;
  qword_5037AF8 = 0;
  qword_5037B00 = 0;
  qword_5037B20 = 0;
  qword_5037B28 = (__int64)&unk_5037B40;
  qword_5037B30 = 1;
  dword_5037B38 = 0;
  byte_5037B3C = 1;
  v63 = sub_C57470();
  v64 = (unsigned int)qword_5037B10;
  v65 = (unsigned int)qword_5037B10 + 1LL;
  if ( v65 > HIDWORD(qword_5037B10) )
  {
    sub_C8D5F0((char *)&unk_5037B18 - 16, &unk_5037B18, v65, 8);
    v64 = (unsigned int)qword_5037B10;
  }
  *(_QWORD *)(qword_5037B08 + 8 * v64) = v63;
  qword_5037B50 = (__int64)&unk_49D9748;
  qword_5037AC0 = (__int64)&unk_49DC090;
  qword_5037B60 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5037B10) = qword_5037B10 + 1;
  qword_5037B80 = (__int64)nullsub_23;
  qword_5037B48 = 0;
  qword_5037B78 = (__int64)sub_984030;
  qword_5037B58 = 0;
  sub_C53080(&qword_5037AC0, "nvptx-disable-combiner-for-O0", 29);
  qword_5037AF0 = 28;
  LOBYTE(qword_5037B48) = 1;
  LOBYTE(dword_5037ACC) = dword_5037ACC & 0x9F | 0x20;
  qword_5037AE8 = (__int64)"Disable DAG combiner for -O0";
  LOWORD(qword_5037B58) = 257;
  sub_C53130(&qword_5037AC0);
  return __cxa_atexit(sub_984900, &qword_5037AC0, &qword_4A427C0);
}

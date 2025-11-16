// Function: sub_28A1B60
// Address: 0x28a1b60
//
__int64 __fastcall sub_28A1B60(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r12
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // r14
  unsigned int v10; // r15d
  unsigned __int64 v11; // rbx
  unsigned int v12; // r12d
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // r8
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // r8
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // r8
  _QWORD *v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r12
  __int64 **v30; // r15
  __int64 (__fastcall *v31)(__int64, unsigned int, __int64, __int64); // rax
  __int64 v32; // rax
  _BYTE *v33; // rax
  __int64 v34; // rax
  __int64 v35; // r10
  __int64 (__fastcall *v36)(__int64, unsigned int, __int64, __int64); // rax
  __int64 v37; // rax
  _BYTE *v38; // r12
  __int64 v39; // rax
  _QWORD *v40; // rax
  _QWORD *v41; // r9
  _QWORD *v42; // rax
  __int64 v43; // rax
  _BYTE *v44; // rax
  _BYTE *v45; // rax
  __int64 v46; // r15
  _QWORD *v47; // rax
  __int64 v48; // r9
  __int64 v49; // r12
  __int64 *v50; // r12
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  _QWORD *v54; // rax
  __int64 v55; // r15
  __int64 v56; // r12
  unsigned __int64 v57; // r8
  unsigned __int64 v58; // rbx
  __int64 v59; // rax
  __int64 v60; // rax
  unsigned int v61; // ecx
  unsigned int v62; // edx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rax
  unsigned __int64 v66; // rdx
  __int64 *v67; // rax
  unsigned __int64 v68; // rbx
  __int64 v69; // rax
  __int64 *v70; // rax
  __int64 v71; // rdx
  __int64 *v72; // rdx
  __int64 v73; // rdx
  __int64 *v74; // rdx
  unsigned __int64 *v75; // rsi
  __int64 v76; // r15
  __int64 v77; // rsi
  char *v78; // r14
  char *v79; // rbx
  unsigned __int64 v80; // rdi
  unsigned __int64 v81; // rdi
  char *v82; // r14
  char *v83; // rbx
  unsigned __int64 v84; // rdi
  unsigned __int64 v85; // rdi
  __int64 v86; // r12
  int v87; // ebx
  __int64 v88; // r12
  int v89; // r8d
  __int64 v90; // rax
  __int64 v91; // [rsp-8h] [rbp-4C8h]
  _BYTE *v92; // [rsp+0h] [rbp-4C0h]
  _QWORD *v93; // [rsp+0h] [rbp-4C0h]
  __int64 v94; // [rsp+0h] [rbp-4C0h]
  unsigned int v95; // [rsp+0h] [rbp-4C0h]
  __int64 v96; // [rsp+0h] [rbp-4C0h]
  unsigned __int64 v97; // [rsp+10h] [rbp-4B0h]
  __int64 v98; // [rsp+10h] [rbp-4B0h]
  int v99; // [rsp+10h] [rbp-4B0h]
  __int64 v100; // [rsp+10h] [rbp-4B0h]
  __int64 v101; // [rsp+20h] [rbp-4A0h]
  _BYTE *v102; // [rsp+20h] [rbp-4A0h]
  unsigned __int8 v103; // [rsp+20h] [rbp-4A0h]
  char v104; // [rsp+20h] [rbp-4A0h]
  __int64 v105; // [rsp+30h] [rbp-490h]
  __int64 v106; // [rsp+30h] [rbp-490h]
  __int64 v107; // [rsp+38h] [rbp-488h]
  __int64 v108; // [rsp+38h] [rbp-488h]
  __int64 v110; // [rsp+48h] [rbp-478h]
  int v111; // [rsp+58h] [rbp-468h]
  __int64 v112; // [rsp+58h] [rbp-468h]
  const char *v113; // [rsp+60h] [rbp-460h] BYREF
  bool v114; // [rsp+68h] [rbp-458h]
  __int16 v115; // [rsp+80h] [rbp-440h]
  __m128i v116[3]; // [rsp+90h] [rbp-430h] BYREF
  __m128i v117[3]; // [rsp+C0h] [rbp-400h] BYREF
  unsigned __int64 *v118; // [rsp+F0h] [rbp-3D0h] BYREF
  __int64 v119; // [rsp+F8h] [rbp-3C8h]
  _BYTE v120[64]; // [rsp+100h] [rbp-3C0h] BYREF
  unsigned int *v121[6]; // [rsp+140h] [rbp-380h] BYREF
  __int64 v122; // [rsp+170h] [rbp-350h]
  __int64 v123; // [rsp+178h] [rbp-348h]
  __int64 v124; // [rsp+180h] [rbp-340h]
  _QWORD *v125; // [rsp+188h] [rbp-338h]
  __int64 v126; // [rsp+190h] [rbp-330h]
  __int64 v127; // [rsp+198h] [rbp-328h]
  __int64 v128; // [rsp+1A0h] [rbp-320h]
  int v129; // [rsp+1A8h] [rbp-318h]
  char *v130; // [rsp+1D0h] [rbp-2F0h] BYREF
  bool v131; // [rsp+1D8h] [rbp-2E8h]
  char *v132; // [rsp+1E0h] [rbp-2E0h] BYREF
  unsigned int v133; // [rsp+1E8h] [rbp-2D8h]
  __int16 v134; // [rsp+1F0h] [rbp-2D0h]
  char v135; // [rsp+300h] [rbp-1C0h] BYREF
  char v136; // [rsp+308h] [rbp-1B8h]
  char *v137; // [rsp+310h] [rbp-1B0h] BYREF
  unsigned int v138; // [rsp+318h] [rbp-1A8h]
  char v139; // [rsp+430h] [rbp-90h] BYREF
  char *v140; // [rsp+438h] [rbp-88h]
  char v141; // [rsp+448h] [rbp-78h] BYREF

  sub_D66630(v116, a3);
  sub_D665A0(v117, a2);
  if ( !(unsigned __int8)sub_CF4E00(a1[4], (__int64)v117, (__int64)v116) )
    return *(_QWORD *)(a2 - 32);
  v7 = *(_QWORD *)(a4 + 40);
  v118 = (unsigned __int64 *)v120;
  v110 = v7;
  v119 = 0x400000000LL;
  v101 = v7 + 48;
  v8 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v101 == v8 )
    goto LABEL_87;
  if ( !v8 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 > 0xA
    || (v111 = sub_B46E30(v8 - 24), v9 = sub_986580(v110), !v111) )
  {
LABEL_87:
    v18 = v110;
  }
  else
  {
    v10 = 0;
    v105 = a4;
    v11 = 4;
    v12 = 0;
    while ( 1 )
    {
      v14 = sub_B46EC0(v9, v10) | 4;
      v15 = v12;
      v16 = v12 + 1LL;
      if ( v16 > v11 )
      {
        v108 = v14;
        sub_C8D5F0((__int64)&v118, v120, v16, 0x10u, v13, v14);
        v15 = (unsigned int)v119;
        v14 = v108;
      }
      v17 = (__int64 *)&v118[2 * v15];
      ++v10;
      *v17 = v110;
      v17[1] = v14;
      v12 = v119 + 1;
      LODWORD(v119) = v119 + 1;
      if ( v111 == v10 )
        break;
      v11 = HIDWORD(v119);
    }
    a4 = v105;
    v18 = *(_QWORD *)(v105 + 40);
  }
  v19 = a1[6];
  v130 = "alias_cont";
  v134 = 259;
  v20 = sub_F36990(v18, (__int64 *)(a4 + 24), 0, 0, v19, 0, (void **)&v130, 0);
  v21 = *(_QWORD *)(a4 + 40);
  v22 = a1[6];
  v112 = v20;
  v130 = "copy";
  v134 = 259;
  v23 = sub_F36990(v21, (__int64 *)(a4 + 24), 0, 0, v22, 0, (void **)&v130, 0);
  v24 = *(_QWORD *)(a4 + 40);
  v25 = a1[6];
  v106 = v23;
  v130 = "no_alias";
  v134 = 259;
  v107 = sub_F36990(v24, (__int64 *)(a4 + 24), 0, 0, v25, 0, (void **)&v130, 0);
  sub_23D0AB0((__int64)v121, a4, 0, 0, 0);
  v26 = (_QWORD *)sub_986580(v110);
  sub_B43D60(v26);
  LOWORD(v124) = 0;
  v122 = v110;
  v123 = v101;
  v27 = sub_B43CC0(a2);
  v28 = sub_AE4420(v27, (__int64)v125, 0);
  v29 = v116[0].m128i_i64[0];
  v30 = (__int64 **)v28;
  v115 = 259;
  v113 = "store.begin";
  if ( v28 == *(_QWORD *)(v116[0].m128i_i64[0] + 8) )
  {
    v102 = (_BYTE *)v116[0].m128i_i64[0];
    goto LABEL_20;
  }
  v31 = *(__int64 (__fastcall **)(__int64, unsigned int, __int64, __int64))(*(_QWORD *)v126 + 120LL);
  if ( (char *)v31 != (char *)sub_920130 )
  {
    v102 = (_BYTE *)v31(v126, 47u, v116[0].m128i_i64[0], (__int64)v30);
    goto LABEL_19;
  }
  if ( *(_BYTE *)v116[0].m128i_i64[0] <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x2Fu) )
      v102 = (_BYTE *)sub_ADAB70(47, v29, v30, 0);
    else
      v102 = (_BYTE *)sub_AA93C0(0x2Fu, v29, (__int64)v30);
LABEL_19:
    if ( v102 )
      goto LABEL_20;
  }
  v134 = 257;
  v86 = sub_B51D30(47, v29, (__int64)v30, (__int64)&v130, 0, 0);
  if ( (unsigned __int8)sub_920620(v86) )
  {
    v87 = v129;
    if ( v128 )
      sub_B99FD0(v86, 3u, v128);
    sub_B45150(v86, v87);
  }
  v102 = (_BYTE *)sub_289B9A0((__int64 *)v121, v86, (__int64)&v113);
LABEL_20:
  v130 = "store.end";
  v134 = 259;
  v113 = (const char *)(v116[0].m128i_i64[1] & 0x3FFFFFFFFFFFFFFFLL);
  v114 = (v116[0].m128i_i64[1] & 0x4000000000000000LL) != 0;
  v32 = sub_CA1930(&v113);
  v33 = (_BYTE *)sub_AD64C0((__int64)v30, v32, 0);
  v34 = sub_929C50(v121, v102, v33, (__int64)&v130, 1u, 1);
  v35 = v117[0].m128i_i64[0];
  v92 = (_BYTE *)v34;
  v113 = "load.begin";
  v115 = 259;
  if ( v30 == *(__int64 ***)(v117[0].m128i_i64[0] + 8) )
  {
    v38 = (_BYTE *)v117[0].m128i_i64[0];
    goto LABEL_27;
  }
  v36 = *(__int64 (__fastcall **)(__int64, unsigned int, __int64, __int64))(*(_QWORD *)v126 + 120LL);
  if ( (char *)v36 != (char *)sub_920130 )
  {
    v100 = v117[0].m128i_i64[0];
    v90 = v36(v126, 47u, v117[0].m128i_i64[0], (__int64)v30);
    v35 = v100;
    v38 = (_BYTE *)v90;
    goto LABEL_26;
  }
  if ( *(_BYTE *)v117[0].m128i_i64[0] <= 0x15u )
  {
    v97 = v117[0].m128i_i64[0];
    if ( (unsigned __int8)sub_AC4810(0x2Fu) )
      v37 = sub_ADAB70(47, v97, v30, 0);
    else
      v37 = sub_AA93C0(0x2Fu, v97, (__int64)v30);
    v35 = v97;
    v38 = (_BYTE *)v37;
LABEL_26:
    if ( v38 )
      goto LABEL_27;
  }
  v134 = 257;
  v88 = sub_B51D30(47, v35, (__int64)v30, (__int64)&v130, 0, 0);
  if ( (unsigned __int8)sub_920620(v88) )
  {
    v89 = v129;
    if ( v128 )
    {
      v99 = v129;
      sub_B99FD0(v88, 3u, v128);
      v89 = v99;
    }
    sub_B45150(v88, v89);
  }
  v38 = (_BYTE *)sub_289B9A0((__int64 *)v121, v88, (__int64)&v113);
LABEL_27:
  v115 = 257;
  v39 = sub_92B530(v121, 0x24u, (__int64)v38, v92, (__int64)&v113);
  v134 = 257;
  v98 = v39;
  v40 = sub_BD2C40(72, 3u);
  v41 = v40;
  if ( v40 )
  {
    v93 = v40;
    sub_B4C9A0((__int64)v40, v112, v107, v98, 3u, (__int64)v40, 0, 0);
    v41 = v93;
  }
  v94 = (__int64)v41;
  (*(void (__fastcall **)(__int64, _QWORD *, char **, __int64, __int64))(*(_QWORD *)v127 + 16LL))(
    v127,
    v41,
    &v130,
    v123,
    v124);
  sub_94AAF0(v121, v94);
  v42 = (_QWORD *)sub_986580(v112);
  sub_B43D60(v42);
  sub_A88F30((__int64)v121, v112, *(_QWORD *)(v112 + 56), 1);
  v130 = "load.end";
  v134 = 259;
  v113 = (const char *)(v117[0].m128i_i64[1] & 0x3FFFFFFFFFFFFFFFLL);
  v114 = (v117[0].m128i_i64[1] & 0x4000000000000000LL) != 0;
  v43 = sub_CA1930(&v113);
  v44 = (_BYTE *)sub_AD64C0((__int64)v30, v43, 0);
  v45 = (_BYTE *)sub_929C50(v121, v38, v44, (__int64)&v130, 1u, 1);
  v115 = 257;
  v46 = sub_92B530(v121, 0x24u, (__int64)v102, v45, (__int64)&v113);
  v134 = 257;
  v47 = sub_BD2C40(72, 3u);
  v49 = (__int64)v47;
  if ( v47 )
  {
    sub_B4C9A0((__int64)v47, v106, v107, v46, 3u, 0, 0, 0);
    v48 = v91;
  }
  (*(void (__fastcall **)(__int64, __int64, char **, __int64, __int64, __int64))(*(_QWORD *)v127 + 16LL))(
    v127,
    v49,
    &v130,
    v123,
    v124,
    v48);
  sub_94AAF0(v121, v49);
  sub_A88F30((__int64)v121, v106, *(_QWORD *)(v106 + 56), 1);
  v50 = sub_BCD420(*(__int64 **)(*(_QWORD *)(a2 + 8) + 24LL), *(unsigned int *)(*(_QWORD *)(a2 + 8) + 32LL));
  v51 = *(_QWORD *)(a2 - 32);
  v115 = 257;
  v52 = *(_QWORD *)(v51 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v52 + 8) - 17 <= 1 )
    v52 = **(_QWORD **)(v52 + 16);
  v95 = *(_DWORD *)(v52 + 8) >> 8;
  v53 = sub_AA4E30(v122);
  v103 = sub_AE5260(v53, (__int64)v50);
  v134 = 257;
  v54 = sub_BD2C40(80, unk_3F10A14);
  v55 = (__int64)v54;
  if ( v54 )
    sub_B4CCA0((__int64)v54, v50, v95, 0, v103, (__int64)&v130, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)v127 + 16LL))(
    v127,
    v55,
    &v113,
    v123,
    v124);
  sub_94AAF0(v121, v55);
  v130 = (char *)(v117[0].m128i_i64[1] & 0x3FFFFFFFFFFFFFFFLL);
  v131 = (v117[0].m128i_i64[1] & 0x4000000000000000LL) != 0;
  v96 = sub_CA1930(&v130);
  v56 = *(_QWORD *)(a2 - 32);
  _BitScanReverse64(&v57, 1LL << (*(_WORD *)(a2 + 2) >> 1));
  v104 = v57 ^ 0x3F;
  _BitScanReverse64(&v58, 1LL << *(_WORD *)(v55 + 2));
  v59 = sub_BCB2E0(v125);
  v60 = sub_ACD640(v59, v96, 0);
  v61 = (unsigned __int8)(63 - (v58 ^ 0x3F));
  v62 = (unsigned __int8)(63 - v104);
  BYTE1(v61) = 1;
  BYTE1(v62) = 1;
  sub_B343C0((__int64)v121, 0xEEu, v55, v61, v56, v62, v60, 0, 0, 0, 0, 0);
  sub_A88F30((__int64)v121, v107, *(_QWORD *)(v107 + 56), 1);
  v134 = 257;
  v5 = sub_D5C860((__int64 *)v121, *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL), 3, (__int64)&v130);
  sub_F0A850(v5, *(_QWORD *)(a2 - 32), v110);
  sub_F0A850(v5, *(_QWORD *)(a2 - 32), v112);
  sub_F0A850(v5, v55, v106);
  v65 = (unsigned int)v119;
  v66 = (unsigned int)v119 + 1LL;
  if ( v66 > HIDWORD(v119) )
  {
    sub_C8D5F0((__int64)&v118, v120, v66, 0x10u, v63, v64);
    v65 = (unsigned int)v119;
  }
  v67 = (__int64 *)&v118[2 * v65];
  v67[1] = v112 & 0xFFFFFFFFFFFFFFFBLL;
  *v67 = v110;
  v68 = v107 & 0xFFFFFFFFFFFFFFFBLL;
  LODWORD(v119) = v119 + 1;
  v69 = (unsigned int)v119;
  if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
  {
    sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 0x10u, v63, v64);
    v69 = (unsigned int)v119;
  }
  v70 = (__int64 *)&v118[2 * v69];
  v70[1] = v68;
  *v70 = v110;
  LODWORD(v119) = v119 + 1;
  v71 = (unsigned int)v119;
  if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
  {
    sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 0x10u, (unsigned int)v119 + 1LL, v64);
    v71 = (unsigned int)v119;
  }
  v72 = (__int64 *)&v118[2 * v71];
  v72[1] = v106 & 0xFFFFFFFFFFFFFFFBLL;
  *v72 = v112;
  LODWORD(v119) = v119 + 1;
  v73 = (unsigned int)v119;
  if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
  {
    sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 0x10u, (unsigned int)v119 + 1LL, v64);
    v73 = (unsigned int)v119;
  }
  v74 = (__int64 *)&v118[2 * v73];
  *v74 = v112;
  v75 = v118;
  v74[1] = v68;
  LODWORD(v119) = v119 + 1;
  v76 = a1[5];
  sub_B26290((__int64)&v130, v75, (unsigned int)v119, 1u);
  v77 = (__int64)&v130;
  sub_B24D40(v76, (__int64)&v130, 0);
  if ( v140 != &v141 )
    _libc_free((unsigned __int64)v140);
  if ( (v136 & 1) != 0 )
  {
    v79 = &v139;
    v78 = (char *)&v137;
  }
  else
  {
    v78 = v137;
    v77 = 72LL * v138;
    if ( !v138 || (v79 = &v137[v77], &v137[v77] == v137) )
    {
LABEL_77:
      sub_C7D6A0((__int64)v78, v77, 8);
      if ( !v131 )
        goto LABEL_57;
      goto LABEL_78;
    }
  }
  do
  {
    if ( *(_QWORD *)v78 != -8192 && *(_QWORD *)v78 != -4096 )
    {
      v80 = *((_QWORD *)v78 + 5);
      if ( (char *)v80 != v78 + 56 )
        _libc_free(v80);
      v81 = *((_QWORD *)v78 + 1);
      if ( (char *)v81 != v78 + 24 )
        _libc_free(v81);
    }
    v78 += 72;
  }
  while ( v79 != v78 );
  if ( (v136 & 1) == 0 )
  {
    v78 = v137;
    v77 = 72LL * v138;
    goto LABEL_77;
  }
  if ( !v131 )
  {
LABEL_57:
    v82 = v132;
    v77 = 72LL * v133;
    if ( !v133 )
      goto LABEL_75;
    v83 = &v132[v77];
    if ( &v132[v77] == v132 )
      goto LABEL_75;
    goto LABEL_59;
  }
LABEL_78:
  v83 = &v135;
  v82 = (char *)&v132;
  do
  {
LABEL_59:
    if ( *(_QWORD *)v82 != -8192 && *(_QWORD *)v82 != -4096 )
    {
      v84 = *((_QWORD *)v82 + 5);
      if ( (char *)v84 != v82 + 56 )
        _libc_free(v84);
      v85 = *((_QWORD *)v82 + 1);
      if ( (char *)v85 != v82 + 24 )
        _libc_free(v85);
    }
    v82 += 72;
  }
  while ( v83 != v82 );
  if ( v131 )
    goto LABEL_67;
  v82 = v132;
  v77 = 72LL * v133;
LABEL_75:
  sub_C7D6A0((__int64)v82, v77, 8);
LABEL_67:
  sub_F94A20(v121, v77);
  if ( v118 != (unsigned __int64 *)v120 )
    _libc_free((unsigned __int64)v118);
  return v5;
}

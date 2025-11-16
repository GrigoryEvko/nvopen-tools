// Function: sub_3796CA0
// Address: 0x3796ca0
//
__int64 __fastcall sub_3796CA0(__int64 *a1, __int64 a2)
{
  unsigned int v2; // ebx
  const __m128i *v4; // rax
  __int64 v5; // rsi
  __m128i v6; // xmm0
  __int64 v7; // rax
  unsigned __int16 v8; // dx
  __int64 v9; // r8
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r8
  int v15; // eax
  _QWORD *v16; // r13
  __int128 v17; // rax
  __int64 v18; // r9
  unsigned int v19; // edx
  unsigned int v20; // ebx
  __int64 v21; // rax
  unsigned __int64 v22; // rsi
  __int64 v23; // rax
  _DWORD *v24; // r10
  __int64 v25; // r11
  __int64 v26; // rdx
  int v27; // edx
  __int64 v28; // r15
  __int64 v29; // r15
  unsigned __int16 v30; // r13
  __int64 v31; // r15
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int16 v34; // ax
  __int64 v35; // rdx
  int v36; // r9d
  unsigned __int16 v37; // cx
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 v40; // r12
  __int64 v41; // rdx
  __int64 v42; // r13
  __int64 v43; // rax
  unsigned __int16 v44; // cx
  __int64 v45; // r8
  int v46; // eax
  __int64 v47; // rdx
  unsigned int v48; // r14d
  __int128 v49; // rcx
  unsigned __int64 v50; // rax
  __int64 v51; // rdx
  __int16 v52; // ax
  __int64 v53; // rdx
  unsigned int v54; // esi
  __int64 v55; // r12
  __int64 v57; // rdx
  bool v58; // al
  __m128i v59; // rax
  unsigned __int64 v60; // r13
  __int8 v61; // r15
  __int64 v62; // rcx
  __int64 v63; // rdx
  __int64 v64; // rax
  unsigned __int64 v65; // rdx
  __int64 v66; // rsi
  unsigned int v67; // edx
  unsigned int v68; // edx
  int v69; // eax
  __int64 v70; // r10
  __int64 v71; // r15
  __int64 v72; // rsi
  unsigned __int8 *v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rsi
  unsigned __int8 *v76; // r8
  __int64 v77; // r9
  _QWORD *v78; // r10
  __int64 v79; // r11
  unsigned int v80; // edx
  __int64 v81; // rsi
  __int64 v82; // rax
  __int64 v83; // rdx
  _QWORD *v84; // r10
  __int64 v85; // r11
  __int64 v86; // r8
  __int64 v87; // r9
  unsigned __int8 *v88; // rax
  unsigned int v89; // edx
  __int64 v90; // rax
  unsigned __int16 *v91; // rdx
  int v92; // eax
  __int64 v93; // r13
  _DWORD *v94; // r13
  unsigned __int16 v95; // cx
  bool v96; // al
  __int64 v97; // rdx
  __int64 v98; // rcx
  __int64 v99; // r8
  bool v100; // al
  unsigned __int16 v101; // si
  __int64 v102; // rdx
  bool v103; // al
  __int128 v104; // [rsp-10h] [rbp-130h]
  __int128 v105; // [rsp-10h] [rbp-130h]
  __int128 v106; // [rsp-10h] [rbp-130h]
  __int64 v107; // [rsp+8h] [rbp-118h]
  _DWORD *v108; // [rsp+10h] [rbp-110h]
  __int64 v109; // [rsp+10h] [rbp-110h]
  __int64 v110; // [rsp+10h] [rbp-110h]
  unsigned __int8 *v111; // [rsp+10h] [rbp-110h]
  __int64 v112; // [rsp+10h] [rbp-110h]
  __int64 v113; // [rsp+10h] [rbp-110h]
  __int64 v114; // [rsp+10h] [rbp-110h]
  __int64 v115; // [rsp+18h] [rbp-108h]
  __int64 (__fastcall *v116)(_DWORD *, __int64, __int64, _QWORD, __int64); // [rsp+20h] [rbp-100h]
  unsigned __int16 v117; // [rsp+20h] [rbp-100h]
  __int64 v118; // [rsp+20h] [rbp-100h]
  _QWORD *v119; // [rsp+20h] [rbp-100h]
  __int64 v120; // [rsp+20h] [rbp-100h]
  __int64 v121; // [rsp+20h] [rbp-100h]
  _DWORD *v122; // [rsp+20h] [rbp-100h]
  _DWORD *v123; // [rsp+20h] [rbp-100h]
  int v124; // [rsp+20h] [rbp-100h]
  __int64 v125; // [rsp+28h] [rbp-F8h]
  __int64 v126; // [rsp+30h] [rbp-F0h]
  _QWORD *v127; // [rsp+30h] [rbp-F0h]
  unsigned __int16 v128; // [rsp+30h] [rbp-F0h]
  __int64 v129; // [rsp+30h] [rbp-F0h]
  _QWORD *v130; // [rsp+30h] [rbp-F0h]
  char v131; // [rsp+30h] [rbp-F0h]
  char v132; // [rsp+30h] [rbp-F0h]
  __int64 v133; // [rsp+38h] [rbp-E8h]
  __int64 v134; // [rsp+40h] [rbp-E0h]
  __int64 v135; // [rsp+58h] [rbp-C8h]
  __int64 v136; // [rsp+58h] [rbp-C8h]
  __int64 v137; // [rsp+58h] [rbp-C8h]
  __int64 v138; // [rsp+60h] [rbp-C0h]
  unsigned __int64 v139; // [rsp+68h] [rbp-B8h]
  __int64 v140; // [rsp+68h] [rbp-B8h]
  unsigned __int16 v141; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v142; // [rsp+78h] [rbp-A8h]
  __int64 v143; // [rsp+80h] [rbp-A0h] BYREF
  int v144; // [rsp+88h] [rbp-98h]
  unsigned int v145; // [rsp+90h] [rbp-90h] BYREF
  __int64 v146; // [rsp+98h] [rbp-88h]
  __int64 v147; // [rsp+A0h] [rbp-80h] BYREF
  int v148; // [rsp+A8h] [rbp-78h]
  __int64 v149; // [rsp+B0h] [rbp-70h]
  __int64 v150; // [rsp+B8h] [rbp-68h]
  __m128i v151; // [rsp+C0h] [rbp-60h] BYREF
  __m128i v152; // [rsp+D0h] [rbp-50h] BYREF

  v4 = *(const __m128i **)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = _mm_loadu_si128(v4);
  v139 = v6.m128i_u64[1];
  v7 = *(_QWORD *)(v4->m128i_i64[0] + 48) + 16LL * v4->m128i_u32[2];
  v8 = *(_WORD *)v7;
  v9 = *(_QWORD *)(v7 + 8);
  v143 = v5;
  v141 = v8;
  v142 = v9;
  if ( v5 )
  {
    sub_B96E90((__int64)&v143, v5, 1);
    v8 = v141;
    v9 = v142;
  }
  v10 = *a1;
  v144 = *(_DWORD *)(a2 + 72);
  sub_2FE6CC0((__int64)&v152, v10, *(_QWORD *)(a1[1] + 64), v8, v9);
  if ( v152.m128i_i8[0] == 5 )
  {
    v136 = sub_37946F0((__int64)a1, v6.m128i_u64[0], v6.m128i_i64[1]);
    v20 = v68;
  }
  else
  {
    if ( v141 )
    {
      v14 = 0;
      LOWORD(v15) = word_4456580[v141 - 1];
    }
    else
    {
      v15 = sub_3009970((__int64)&v141, v10, v11, v12, v13);
      HIWORD(v2) = HIWORD(v15);
      v14 = v57;
    }
    v16 = (_QWORD *)a1[1];
    LOWORD(v2) = v15;
    v135 = v14;
    *(_QWORD *)&v17 = sub_3400EE0((__int64)v16, 0, (__int64)&v143, 0, v6);
    v136 = (__int64)sub_3406EB0(v16, 0x9Eu, (__int64)&v143, v2, v135, v18, *(_OWORD *)&v6, v17);
    v20 = v19;
  }
  v21 = *(_QWORD *)(a2 + 40);
  v22 = *(_QWORD *)(v21 + 40);
  v23 = sub_37946F0((__int64)a1, v22, *(_QWORD *)(v21 + 48));
  v24 = (_DWORD *)*a1;
  v25 = v20;
  v133 = v26;
  v134 = v23;
  v27 = *(_DWORD *)(*a1 + 60);
  v28 = 16LL * v20;
  if ( *(_DWORD *)(*a1 + 64) == v27 )
  {
    v69 = v24[17];
    goto LABEL_37;
  }
  if ( *(_DWORD *)(v136 + 24) != 208 )
  {
    v29 = *(_QWORD *)(v136 + 48) + v28;
    v30 = *(_WORD *)v29;
    v31 = *(_QWORD *)(v29 + 8);
    goto LABEL_10;
  }
  v91 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(v136 + 40) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(v136 + 40) + 8LL));
  v92 = *v91;
  v93 = *((_QWORD *)v91 + 1);
  v151.m128i_i16[0] = v92;
  v151.m128i_i64[1] = v93;
  if ( (_WORD)v92 )
  {
    if ( (unsigned __int16)(v92 - 17) > 0xD3u )
    {
      v152.m128i_i16[0] = v92;
      v152.m128i_i64[1] = v93;
      v94 = v24;
      goto LABEL_64;
    }
    v94 = v24;
    LOWORD(v92) = word_4456580[v92 - 1];
    v102 = 0;
  }
  else
  {
    v122 = v24;
    v96 = sub_30070B0((__int64)&v151);
    v24 = v122;
    v25 = v20;
    if ( !v96 )
    {
      v152.m128i_i64[1] = v93;
      v94 = v122;
      v152.m128i_i16[0] = 0;
      goto LABEL_70;
    }
    LOWORD(v92) = sub_3009970((__int64)&v151, v22, v97, v98, v99);
    v94 = (_DWORD *)*a1;
    v25 = v20;
    v24 = v122;
  }
  v152.m128i_i16[0] = v92;
  v152.m128i_i64[1] = v102;
  if ( !(_WORD)v92 )
  {
LABEL_70:
    v113 = v25;
    v123 = v24;
    v131 = sub_3007030((__int64)&v152);
    v100 = sub_30070B0((__int64)&v152);
    v24 = v123;
    v25 = v113;
    if ( v100 )
      goto LABEL_67;
    if ( !v131 )
      goto LABEL_72;
    goto LABEL_82;
  }
LABEL_64:
  v95 = v92 - 17;
  if ( (unsigned __int16)(v92 - 10) > 6u && (unsigned __int16)(v92 - 126) > 0x31u )
  {
    if ( v95 <= 0xD3u )
    {
LABEL_67:
      v27 = v24[17];
      goto LABEL_73;
    }
LABEL_72:
    v27 = v24[15];
    goto LABEL_73;
  }
  if ( v95 <= 0xD3u )
    goto LABEL_67;
LABEL_82:
  v27 = v24[16];
LABEL_73:
  v152 = _mm_loadu_si128(&v151);
  if ( !v151.m128i_i16[0] )
  {
    v114 = v25;
    v124 = v27;
    v132 = sub_3007030((__int64)&v152);
    v103 = sub_30070B0((__int64)&v152);
    v27 = v124;
    v25 = v114;
    if ( v103 )
      goto LABEL_87;
    if ( !v132 )
      goto LABEL_77;
LABEL_85:
    v69 = v94[16];
    goto LABEL_37;
  }
  v101 = v151.m128i_i16[0] - 17;
  if ( (unsigned __int16)(v151.m128i_i16[0] - 10) > 6u && (unsigned __int16)(v151.m128i_i16[0] - 126) > 0x31u )
  {
    if ( v101 > 0xD3u )
    {
LABEL_77:
      v69 = v94[15];
      goto LABEL_37;
    }
    goto LABEL_87;
  }
  if ( v101 > 0xD3u )
    goto LABEL_85;
LABEL_87:
  v69 = v94[17];
LABEL_37:
  v70 = a1[1];
  v71 = *(_QWORD *)(v136 + 48) + v28;
  v30 = *(_WORD *)v71;
  v31 = *(_QWORD *)(v71 + 8);
  if ( v27 != v69 )
  {
    if ( v27 == 1 )
    {
      v72 = *(_QWORD *)(a2 + 80);
      v152.m128i_i64[0] = v72;
      if ( v72 )
      {
        v118 = v25;
        v129 = v70;
        sub_B96E90((__int64)&v152, v72, 1);
        v25 = v118;
        v70 = v129;
      }
      v152.m128i_i32[2] = *(_DWORD *)(a2 + 72);
      v110 = v25;
      v119 = (_QWORD *)v70;
      v73 = sub_3400BD0(v70, 1, (__int64)&v152, v30, v31, 0, v6, 0);
      v75 = *(_QWORD *)(a2 + 80);
      v76 = v73;
      v77 = v74;
      v151.m128i_i64[0] = v75;
      v78 = v119;
      v79 = v110;
      if ( v75 )
      {
        v115 = v74;
        v107 = v110;
        v111 = v73;
        sub_B96E90((__int64)&v151, v75, 1);
        v79 = v107;
        v76 = v111;
        v77 = v115;
        v78 = v119;
      }
      *((_QWORD *)&v105 + 1) = v77;
      *(_QWORD *)&v105 = v76;
      v151.m128i_i32[2] = *(_DWORD *)(a2 + 72);
      v139 = v79 | v6.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v136 = (__int64)sub_3406EB0(v78, 0xBAu, (__int64)&v151, v30, v31, v77, __PAIR128__(v139, v136), v105);
      v20 = v80;
      if ( v151.m128i_i64[0] )
        sub_B91220((__int64)&v151, v151.m128i_i64[0]);
      v81 = v152.m128i_i64[0];
      if ( !v152.m128i_i64[0] )
        goto LABEL_47;
    }
    else
    {
      if ( v27 != 2 )
        goto LABEL_47;
      v120 = v25;
      v130 = (_QWORD *)a1[1];
      v82 = sub_33F7D60(v130, 2, 0);
      v84 = v130;
      v85 = v120;
      v86 = v82;
      v87 = v83;
      v152.m128i_i64[0] = *(_QWORD *)(a2 + 80);
      if ( v152.m128i_i64[0] )
      {
        v125 = v83;
        v112 = v120;
        v121 = v82;
        sub_B96E90((__int64)&v152, v152.m128i_i64[0], 1);
        v85 = v112;
        v86 = v121;
        v87 = v125;
        v84 = v130;
      }
      *((_QWORD *)&v106 + 1) = v87;
      *(_QWORD *)&v106 = v86;
      v152.m128i_i32[2] = *(_DWORD *)(a2 + 72);
      v139 = v85 | v6.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v88 = sub_3406EB0(v84, 0xDEu, (__int64)&v152, v30, v31, v87, __PAIR128__(v139, v136), v106);
      v81 = v152.m128i_i64[0];
      v136 = (__int64)v88;
      v20 = v89;
      if ( !v152.m128i_i64[0] )
        goto LABEL_47;
    }
    sub_B91220((__int64)&v152, v81);
  }
LABEL_47:
  v24 = (_DWORD *)*a1;
LABEL_10:
  v32 = a1[1];
  v108 = v24;
  v116 = *(__int64 (__fastcall **)(_DWORD *, __int64, __int64, _QWORD, __int64))(*(_QWORD *)v24 + 528LL);
  v126 = *(_QWORD *)(v32 + 64);
  v33 = sub_2E79000(*(__int64 **)(v32 + 40));
  v34 = v116(v108, v33, v126, v30, v31);
  LOWORD(v145) = v34;
  v37 = v34;
  v146 = v35;
  if ( v34 == v30 )
  {
    if ( v30 || v35 == v31 )
    {
LABEL_12:
      v127 = (_QWORD *)a1[1];
      goto LABEL_13;
    }
    v152.m128i_i64[1] = v31;
    v152.m128i_i16[0] = 0;
    goto LABEL_25;
  }
  v152.m128i_i16[0] = v30;
  v152.m128i_i64[1] = v31;
  if ( !v30 )
  {
LABEL_25:
    v128 = v34;
    v59.m128i_i64[0] = sub_3007260((__int64)&v152);
    v37 = v128;
    v151 = v59;
    v60 = v59.m128i_i64[0];
    v61 = v59.m128i_i8[8];
    goto LABEL_26;
  }
  if ( v30 == 1 || (unsigned __int16)(v30 - 504) <= 7u )
    goto LABEL_89;
  v90 = 16LL * (v30 - 1) + 71615648;
  v60 = *(_QWORD *)&byte_444C4A0[16 * v30 - 16];
  v61 = *(_BYTE *)(v90 + 8);
LABEL_26:
  if ( !v37 )
  {
    v62 = sub_3007260((__int64)&v145);
    v64 = v63;
    v149 = v62;
    v65 = v62;
    v150 = v64;
    goto LABEL_28;
  }
  if ( v37 == 1 || (unsigned __int16)(v37 - 504) <= 7u )
LABEL_89:
    BUG();
  v65 = *(_QWORD *)&byte_444C4A0[16 * v37 - 16];
  LOBYTE(v64) = byte_444C4A0[16 * v37 - 8];
LABEL_28:
  v127 = (_QWORD *)a1[1];
  if ( (!(_BYTE)v64 || v61) && v65 < v60 )
  {
    v66 = *(_QWORD *)(a2 + 80);
    v152.m128i_i64[0] = v66;
    if ( v66 )
      sub_B96E90((__int64)&v152, v66, 1);
    v152.m128i_i32[2] = *(_DWORD *)(a2 + 72);
    v139 = v20 | v139 & 0xFFFFFFFF00000000LL;
    v136 = (__int64)sub_33FAF80((__int64)v127, 216, (__int64)&v152, v145, v146, v36, v6);
    v20 = v67;
    if ( v152.m128i_i64[0] )
      sub_B91220((__int64)&v152, v152.m128i_i64[0]);
    goto LABEL_12;
  }
LABEL_13:
  v38 = sub_37946F0((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL));
  v39 = *(_QWORD *)(a2 + 80);
  v40 = v38;
  v42 = v41;
  v43 = *(_QWORD *)(v134 + 48) + 16LL * (unsigned int)v133;
  v44 = *(_WORD *)v43;
  v45 = *(_QWORD *)(v43 + 8);
  v147 = v39;
  if ( v39 )
  {
    v109 = v45;
    v117 = v44;
    sub_B96E90((__int64)&v147, v39, 1);
    v45 = v109;
    v44 = v117;
  }
  v46 = *(_DWORD *)(a2 + 72);
  v47 = v20;
  v48 = v44;
  *((_QWORD *)&v49 + 1) = v133;
  v148 = v46;
  v138 = v136;
  *(_QWORD *)&v49 = v134;
  v50 = v47 | v139 & 0xFFFFFFFF00000000LL;
  v51 = *(_QWORD *)(v136 + 48) + 16 * v47;
  v140 = v50;
  v52 = *(_WORD *)v51;
  v53 = *(_QWORD *)(v51 + 8);
  v152.m128i_i16[0] = v52;
  v152.m128i_i64[1] = v53;
  if ( v52 )
  {
    v54 = ((unsigned __int16)(v52 - 17) < 0xD4u) + 205;
  }
  else
  {
    v137 = v45;
    v58 = sub_30070B0((__int64)&v152);
    *(_QWORD *)&v49 = v134;
    *((_QWORD *)&v49 + 1) = v133;
    v45 = v137;
    v54 = 205 - (!v58 - 1);
  }
  *((_QWORD *)&v104 + 1) = v42;
  *(_QWORD *)&v104 = v40;
  v55 = sub_340EC60(v127, v54, (__int64)&v147, v48, v45, 0, v138, v140, v49, v104);
  if ( v147 )
    sub_B91220((__int64)&v147, v147);
  if ( v143 )
    sub_B91220((__int64)&v143, v143);
  return v55;
}

// Function: sub_21C2FA0
// Address: 0x21c2fa0
//
__int64 __fastcall sub_21C2FA0(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v7; // rsi
  unsigned __int16 v8; // ax
  unsigned int v9; // r13d
  __int64 v10; // r8
  __int64 v11; // rdx
  char v12; // r14
  __int64 v13; // rdi
  __int64 v14; // r12
  unsigned int v15; // eax
  int v16; // eax
  __int64 v17; // r8
  int v18; // r12d
  __int64 v19; // rdx
  unsigned int v20; // eax
  __int64 v21; // rax
  __int64 v22; // rdx
  int v23; // ecx
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 *v26; // rax
  __int64 v27; // r14
  __int64 v28; // rdx
  unsigned __int8 *v29; // rax
  int v30; // r8d
  int v31; // ecx
  int v32; // edx
  __int16 v33; // r11
  __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rdi
  int v37; // edx
  __int64 v38; // rax
  __int64 v39; // rdi
  int v40; // edx
  __int64 v41; // rax
  __int64 v42; // rdi
  int v43; // edx
  __int64 v44; // rax
  __int64 v45; // rdi
  int v46; // edx
  int v47; // edx
  _QWORD *v48; // rdi
  __int64 v49; // r9
  __int64 v50; // r12
  int v52; // r8d
  int v53; // ecx
  int v54; // edx
  __int16 v55; // r12
  __int64 v56; // rdi
  __int64 v57; // rax
  __int64 v58; // rdi
  int v59; // edx
  __int64 v60; // rax
  __int64 v61; // rdi
  int v62; // edx
  __int64 v63; // rax
  __int64 v64; // rdi
  int v65; // edx
  __int64 v66; // rax
  __int64 v67; // rdi
  int v68; // edx
  __int64 v69; // rax
  __m128i v70; // xmm3
  int v71; // edx
  __m128i v72; // xmm4
  _QWORD *v73; // rdi
  __int64 v74; // r9
  __int16 v75; // r10
  __int64 v76; // rdi
  __int64 v77; // rax
  __int64 v78; // rdi
  int v79; // edx
  __int64 v80; // rax
  __int64 v81; // rdi
  int v82; // edx
  __int64 v83; // rax
  __int64 v84; // rdi
  int v85; // edx
  __int64 v86; // rax
  __int64 v87; // rdi
  int v88; // edx
  __int64 v89; // rax
  __int64 v90; // r9
  __m128i v91; // xmm1
  __m128i v92; // xmm2
  int v93; // edx
  _QWORD *v94; // rdi
  __int64 v95; // rdx
  __int16 v96; // r10
  __int64 v97; // rdi
  __int64 v98; // rax
  __int64 v99; // rdi
  int v100; // edx
  __int64 v101; // rax
  __int64 v102; // rdi
  int v103; // edx
  __int64 v104; // rax
  __int64 v105; // rdi
  int v106; // edx
  __int64 v107; // rax
  __int64 v108; // rdi
  int v109; // edx
  __int64 v110; // rax
  __m128i v111; // xmm0
  int v112; // edx
  _QWORD *v113; // rax
  __int64 v114; // rdx
  __int64 v115; // rcx
  __int64 v116; // r8
  __int64 v117; // r9
  __int128 v118; // [rsp-10h] [rbp-190h]
  __int128 v119; // [rsp-10h] [rbp-190h]
  __int128 v120; // [rsp-10h] [rbp-190h]
  int v121; // [rsp+0h] [rbp-180h]
  int v122; // [rsp+0h] [rbp-180h]
  __int64 v123; // [rsp+20h] [rbp-160h]
  __int32 v124; // [rsp+28h] [rbp-158h]
  int v125; // [rsp+2Ch] [rbp-154h]
  __int64 v126; // [rsp+30h] [rbp-150h]
  bool v127; // [rsp+3Bh] [rbp-145h]
  unsigned int v128; // [rsp+3Ch] [rbp-144h]
  unsigned __int8 v129; // [rsp+40h] [rbp-140h]
  unsigned __int8 v130; // [rsp+48h] [rbp-138h]
  unsigned int v131; // [rsp+48h] [rbp-138h]
  unsigned int v132; // [rsp+4Ch] [rbp-134h]
  __int64 v133; // [rsp+50h] [rbp-130h]
  __int16 v134; // [rsp+50h] [rbp-130h]
  __int64 v135; // [rsp+58h] [rbp-128h]
  __int64 v136; // [rsp+58h] [rbp-128h]
  __int16 v137; // [rsp+58h] [rbp-128h]
  __int64 v138; // [rsp+60h] [rbp-120h] BYREF
  __int64 v139; // [rsp+68h] [rbp-118h] BYREF
  __int64 v140; // [rsp+70h] [rbp-110h] BYREF
  __int64 v141; // [rsp+78h] [rbp-108h] BYREF
  __int64 v142; // [rsp+80h] [rbp-100h] BYREF
  int v143; // [rsp+88h] [rbp-F8h]
  __m128i v144; // [rsp+90h] [rbp-F0h] BYREF
  __m128i v145; // [rsp+A0h] [rbp-E0h] BYREF
  __m128i v146; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v147; // [rsp+C0h] [rbp-C0h] BYREF
  int v148; // [rsp+C8h] [rbp-B8h]
  __int64 v149; // [rsp+D0h] [rbp-B0h]
  int v150; // [rsp+D8h] [rbp-A8h]
  __int64 v151; // [rsp+E0h] [rbp-A0h]
  int v152; // [rsp+E8h] [rbp-98h]
  __int64 v153; // [rsp+F0h] [rbp-90h]
  int v154; // [rsp+F8h] [rbp-88h]
  __int64 v155; // [rsp+100h] [rbp-80h]
  int v156; // [rsp+108h] [rbp-78h]
  __int64 v157; // [rsp+110h] [rbp-70h]
  int v158; // [rsp+118h] [rbp-68h]
  __m128i v159; // [rsp+120h] [rbp-60h]
  __m128i v160; // [rsp+130h] [rbp-50h]
  __int64 v161; // [rsp+140h] [rbp-40h]
  __int32 v162; // [rsp+148h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 72);
  v142 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v142, v7, 2);
  v143 = *(_DWORD *)(a2 + 64);
  v8 = *(_WORD *)(a2 + 24);
  if ( v8 == 186 )
  {
    v9 = 0;
    if ( (*(_WORD *)(a2 + 26) & 0x380) != 0 )
      goto LABEL_26;
    v10 = a2;
    v11 = 0;
  }
  else
  {
    if ( v8 == 221 || (unsigned int)v8 - 222 <= 0xC || (unsigned int)v8 - 219 <= 1 )
      v11 = a2;
    else
      v11 = 0;
    v10 = 0;
  }
  v12 = *(_BYTE *)(a2 + 88);
  v133 = v11;
  v9 = 0;
  v135 = v10;
  if ( v12 )
  {
    v13 = *(_QWORD *)(a2 + 104);
    v130 = *(_BYTE *)(v13 + 37) & 0xF;
    v9 = (unsigned __int8)byte_42880A0[8 * v130 + 2];
    if ( (_BYTE)v9 )
      goto LABEL_25;
    v132 = sub_21BD7A0((_QWORD *)v13);
    v14 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 32LL));
    v15 = sub_1E340A0(*(_QWORD *)(a2 + 104));
    v16 = sub_15A9520(v14, v15);
    v17 = v135;
    v18 = v16;
    v19 = v133;
    if ( v132 == 3 || (v127 = 0, v132 <= 1) )
      v127 = v130 == 2 || (*(_BYTE *)(a2 + 26) & 8) != 0;
    if ( (unsigned __int8)(v12 - 14) <= 0x5Fu )
    {
      v131 = 0;
      v128 = 32;
      switch ( v12 )
      {
        case 'V':
        case 'W':
        case 'X':
        case 'b':
        case 'c':
        case 'd':
          goto LABEL_37;
        case 'Y':
        case 'Z':
        case '[':
        case '\\':
        case ']':
        case '^':
        case '_':
        case '`':
        case 'a':
        case 'e':
        case 'f':
        case 'g':
        case 'h':
        case 'i':
        case 'j':
        case 'k':
        case 'l':
        case 'm':
          v131 = 2;
          break;
        default:
          break;
      }
    }
    else
    {
      v20 = sub_21BD810(v12);
      v19 = v133;
      v128 = v20;
      if ( (unsigned __int8)(v12 - 86) <= 0x17u || (v131 = 0, (unsigned __int8)(v12 - 8) <= 5u) )
      {
        v131 = 2;
        if ( v12 == 8 )
LABEL_37:
          v131 = 3;
      }
    }
    v21 = *(_QWORD *)(a2 + 32);
    v126 = *(_QWORD *)v21;
    v124 = *(_DWORD *)(v21 + 8);
    if ( v17 )
      v22 = *(_QWORD *)(v17 + 32) + 40LL;
    else
      v22 = *(_QWORD *)(v19 + 32) + 80LL;
    v23 = *(_DWORD *)(v22 + 8);
    v24 = *(_QWORD *)v22;
    v25 = 80;
    v125 = v23;
    if ( *(_WORD *)(a2 + 24) != 186 )
      v25 = 40;
    v123 = v24;
    v26 = (__int64 *)(v25 + v21);
    v27 = *v26;
    v28 = v26[1];
    v144.m128i_i64[0] = 0;
    v144.m128i_i32[2] = 0;
    v29 = *(unsigned __int8 **)(v24 + 40);
    v145.m128i_i64[0] = 0;
    v145.m128i_i32[2] = 0;
    v146.m128i_i64[0] = 0;
    v146.m128i_i32[2] = 0;
    v136 = v28;
    v129 = *v29;
    if ( sub_21C2A00(a1, v27, v28, (__int64)&v144) )
    {
      v141 = 0x100000E11LL;
      v140 = 0x100000E05LL;
      v139 = 0x100000DFFLL;
      v138 = 0x100000E23LL;
      sub_21BD570(
        (__int64)&v147,
        v129,
        3625,
        3607,
        3613,
        (__int64)&v138,
        (__int64)&v139,
        (__int64)&v140,
        3595,
        (__int64)&v141);
      if ( !BYTE4(v147) )
        goto LABEL_25;
      v96 = v147;
      v97 = *(_QWORD *)(a1 + 272);
      v147 = v123;
      v137 = v96;
      v148 = v125;
      v98 = sub_1D38BB0(v97, v127, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
      v99 = *(_QWORD *)(a1 + 272);
      v150 = v100;
      v149 = v98;
      v101 = sub_1D38BB0(v99, v132, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
      v102 = *(_QWORD *)(a1 + 272);
      v152 = v103;
      v151 = v101;
      v104 = sub_1D38BB0(v102, 1, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
      v105 = *(_QWORD *)(a1 + 272);
      v154 = v106;
      v153 = v104;
      v107 = sub_1D38BB0(v105, v131, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
      v108 = *(_QWORD *)(a1 + 272);
      v156 = v109;
      v155 = v107;
      v110 = sub_1D38BB0(v108, v128, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
      v111 = _mm_loadu_si128(&v144);
      v94 = *(_QWORD **)(a1 + 272);
      v157 = v110;
      v158 = v112;
      v95 = 8;
      v160.m128i_i64[0] = v126;
      v159 = v111;
      v160.m128i_i32[2] = v124;
    }
    else
    {
      if ( 8 * v18 == 64 )
      {
        if ( !(unsigned __int8)sub_21C2BC0(a1, v27, v27, v136, (__int64)&v146, (__int64)&v145, a3, a4, a5) )
        {
          if ( !(unsigned __int8)sub_21C2F80(a1, v27, v27, v136, (__int64)&v146, (__int64)&v145, a3, a4, a5) )
          {
            v30 = 3609;
            v31 = 3603;
            v32 = 3621;
            v141 = 0x100000E0DLL;
            v140 = 0x100000E01LL;
            v139 = 0x100000DFBLL;
            v138 = 0x100000E1FLL;
            v121 = 3591;
LABEL_22:
            sub_21BD570(
              (__int64)&v147,
              v129,
              v32,
              v31,
              v30,
              (__int64)&v138,
              (__int64)&v139,
              (__int64)&v140,
              v121,
              (__int64)&v141);
            if ( BYTE4(v147) )
            {
              v33 = v147;
              v147 = v24;
              v34 = *(_QWORD *)(a1 + 272);
              v134 = v33;
              v148 = v125;
              v35 = sub_1D38BB0(v34, v127, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
              v36 = *(_QWORD *)(a1 + 272);
              v150 = v37;
              v149 = v35;
              v38 = sub_1D38BB0(v36, v132, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
              v39 = *(_QWORD *)(a1 + 272);
              v152 = v40;
              v151 = v38;
              v41 = sub_1D38BB0(v39, 1, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
              v42 = *(_QWORD *)(a1 + 272);
              v154 = v43;
              v153 = v41;
              v44 = sub_1D38BB0(v42, v131, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
              v45 = *(_QWORD *)(a1 + 272);
              v156 = v46;
              v155 = v44;
              v157 = sub_1D38BB0(v45, v128, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
              v158 = v47;
              v48 = *(_QWORD **)(a1 + 272);
              v159.m128i_i64[1] = v136;
              *((_QWORD *)&v118 + 1) = 8;
              *(_QWORD *)&v118 = &v147;
              v160.m128i_i64[0] = v126;
              v159.m128i_i64[0] = v27;
              v160.m128i_i32[2] = v124;
              v50 = sub_1D2CDB0(v48, v134, (__int64)&v142, 1, 0, v49, v118);
              goto LABEL_48;
            }
LABEL_25:
            v9 = 0;
            goto LABEL_26;
          }
          v52 = 3611;
          v53 = 3605;
          v54 = 3623;
          v141 = 0x100000E0FLL;
          v140 = 0x100000E03LL;
          v139 = 0x100000DFDLL;
          v138 = 0x100000E21LL;
          v122 = 3593;
          goto LABEL_42;
        }
      }
      else if ( !(unsigned __int8)sub_21C2BA0(a1, v27, v27, v136, (__int64)&v146, (__int64)&v145, a3, a4, a5) )
      {
        if ( !(unsigned __int8)sub_21C2F60(a1, v27, v27, v136, (__int64)&v146, (__int64)&v145, a3, a4, a5) )
        {
          v30 = 3608;
          v31 = 3602;
          v32 = 3620;
          v141 = 0x100000E0CLL;
          v140 = 0x100000E00LL;
          v139 = 0x100000DFALL;
          v138 = 0x100000E1ELL;
          v121 = 3590;
          goto LABEL_22;
        }
        v52 = 3610;
        v53 = 3604;
        v54 = 3622;
        v141 = 0x100000E0ELL;
        v140 = 0x100000E02LL;
        v139 = 0x100000DFCLL;
        v138 = 0x100000E20LL;
        v122 = 3592;
LABEL_42:
        sub_21BD570(
          (__int64)&v147,
          v129,
          v54,
          v53,
          v52,
          (__int64)&v138,
          (__int64)&v139,
          (__int64)&v140,
          v122,
          (__int64)&v141);
        if ( !BYTE4(v147) )
          goto LABEL_25;
        v55 = v147;
        v56 = *(_QWORD *)(a1 + 272);
        v147 = v123;
        v148 = v125;
        v57 = sub_1D38BB0(v56, v127, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
        v58 = *(_QWORD *)(a1 + 272);
        v150 = v59;
        v149 = v57;
        v60 = sub_1D38BB0(v58, v132, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
        v61 = *(_QWORD *)(a1 + 272);
        v152 = v62;
        v151 = v60;
        v63 = sub_1D38BB0(v61, 1, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
        v64 = *(_QWORD *)(a1 + 272);
        v154 = v65;
        v153 = v63;
        v66 = sub_1D38BB0(v64, v131, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
        v67 = *(_QWORD *)(a1 + 272);
        v156 = v68;
        v155 = v66;
        v69 = sub_1D38BB0(v67, v128, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
        v70 = _mm_loadu_si128(&v146);
        v157 = v69;
        v158 = v71;
        v72 = _mm_loadu_si128(&v145);
        *((_QWORD *)&v119 + 1) = 9;
        v73 = *(_QWORD **)(a1 + 272);
        *(_QWORD *)&v119 = &v147;
        v161 = v126;
        v159 = v70;
        v162 = v124;
        v160 = v72;
        v50 = sub_1D2CDB0(v73, v55, (__int64)&v142, 1, 0, v74, v119);
LABEL_48:
        if ( v50 )
        {
          v9 = 1;
          v113 = (_QWORD *)sub_1E0A240(*(_QWORD *)(a1 + 256), 1);
          *v113 = *(_QWORD *)(a2 + 104);
          *(_QWORD *)(v50 + 88) = v113;
          *(_QWORD *)(v50 + 96) = v113 + 1;
          sub_1D444E0(*(_QWORD *)(a1 + 272), a2, v50);
          sub_1D49010(v50);
          sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v114, v115, v116, v117);
        }
        goto LABEL_26;
      }
      v141 = 0x100000E10LL;
      v140 = 0x100000E04LL;
      v139 = 0x100000DFELL;
      v138 = 0x100000E22LL;
      sub_21BD570(
        (__int64)&v147,
        v129,
        3624,
        3606,
        3612,
        (__int64)&v138,
        (__int64)&v139,
        (__int64)&v140,
        3594,
        (__int64)&v141);
      if ( !BYTE4(v147) )
        goto LABEL_25;
      v75 = v147;
      v76 = *(_QWORD *)(a1 + 272);
      v147 = v123;
      v137 = v75;
      v148 = v125;
      v77 = sub_1D38BB0(v76, v127, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
      v78 = *(_QWORD *)(a1 + 272);
      v150 = v79;
      v149 = v77;
      v80 = sub_1D38BB0(v78, v132, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
      v81 = *(_QWORD *)(a1 + 272);
      v152 = v82;
      v151 = v80;
      v83 = sub_1D38BB0(v81, 1, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
      v84 = *(_QWORD *)(a1 + 272);
      v154 = v85;
      v153 = v83;
      v86 = sub_1D38BB0(v84, v131, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
      v87 = *(_QWORD *)(a1 + 272);
      v156 = v88;
      v155 = v86;
      v89 = sub_1D38BB0(v87, v128, (__int64)&v142, 5, 0, 1, a3, a4, a5, 0);
      v91 = _mm_loadu_si128(&v146);
      v92 = _mm_loadu_si128(&v145);
      v157 = v89;
      v158 = v93;
      v94 = *(_QWORD **)(a1 + 272);
      v95 = 9;
      v161 = v126;
      v159 = v91;
      v162 = v124;
      v160 = v92;
    }
    *((_QWORD *)&v120 + 1) = v95;
    *(_QWORD *)&v120 = &v147;
    v50 = sub_1D2CDB0(v94, v137, (__int64)&v142, 1, 0, v90, v120);
    goto LABEL_48;
  }
LABEL_26:
  if ( v142 )
    sub_161E7C0((__int64)&v142, v142);
  return v9;
}

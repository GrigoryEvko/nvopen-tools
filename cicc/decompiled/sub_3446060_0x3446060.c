// Function: sub_3446060
// Address: 0x3446060
//
unsigned __int8 *__fastcall sub_3446060(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        unsigned int a8)
{
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned int v12; // r14d
  bool v13; // r14
  __int64 v14; // rax
  unsigned __int8 *result; // rax
  __int64 *v16; // rax
  __int64 v17; // r15
  __int64 v18; // r9
  __int64 v19; // r11
  unsigned int v20; // r10d
  bool v21; // r13
  __int64 v22; // r15
  unsigned int v23; // eax
  unsigned int v24; // r11d
  int v25; // r11d
  unsigned __int64 v26; // r11
  unsigned int v27; // r12d
  int v28; // eax
  unsigned __int64 v29; // r12
  unsigned int v30; // eax
  __int64 v31; // rsi
  __int16 v32; // dx
  unsigned int v33; // eax
  int v34; // eax
  int v35; // ecx
  __int16 v36; // ax
  __int64 v37; // rdx
  unsigned __int64 v38; // r12
  __int64 v39; // rax
  __int64 v40; // rcx
  __int64 v41; // rsi
  __int64 v42; // rdx
  __int64 v43; // r8
  __int64 v45; // rax
  __int64 v46; // rcx
  __int64 v47; // rsi
  __int64 v48; // rdx
  __int64 v49; // r8
  int v50; // eax
  __int64 v51; // rax
  int v52; // eax
  int v53; // eax
  __int64 v54; // rsi
  unsigned __int8 *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // r13
  unsigned __int8 *v58; // r12
  __int128 v59; // rax
  __int64 v60; // r9
  unsigned __int8 *v61; // rax
  unsigned int v62; // edx
  unsigned __int8 *v63; // rcx
  unsigned int v64; // eax
  unsigned __int64 v65; // rdi
  unsigned __int64 v66; // rdi
  __int64 v67; // rax
  __int64 v68; // rax
  unsigned int v69; // r10d
  __int64 v70; // r9
  __int64 v71; // rax
  unsigned int v72; // r13d
  int v73; // eax
  __int64 v74; // rax
  __int64 v75; // rax
  unsigned int v76; // r13d
  int v77; // eax
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // r9
  unsigned int v81; // r10d
  __int64 v82; // rax
  unsigned int v83; // r13d
  int v84; // eax
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // r9
  unsigned int v88; // r15d
  int v89; // eax
  unsigned __int64 v90; // rsi
  unsigned __int64 v91; // rax
  char v92; // cl
  __int64 *v93; // rdx
  unsigned __int32 v94; // r12d
  __int16 v95; // ax
  __int64 v96; // r8
  __int64 v97; // r9
  __int64 v98; // rcx
  unsigned __int8 *v99; // rax
  __int64 v100; // rdx
  __int64 v101; // r13
  unsigned __int8 *v102; // r12
  __int128 v103; // rax
  __int64 v104; // r9
  unsigned __int8 *v105; // rax
  unsigned int v106; // edx
  __int64 v107; // rdx
  __int128 v108; // [rsp-10h] [rbp-120h]
  __int128 v109; // [rsp-10h] [rbp-120h]
  unsigned int v110; // [rsp+8h] [rbp-108h]
  __int64 v111; // [rsp+8h] [rbp-108h]
  __int64 v112; // [rsp+10h] [rbp-100h]
  unsigned int v113; // [rsp+10h] [rbp-100h]
  __int64 v114; // [rsp+18h] [rbp-F8h]
  __int64 v115; // [rsp+18h] [rbp-F8h]
  __int64 v116; // [rsp+20h] [rbp-F0h]
  unsigned int v117; // [rsp+2Ch] [rbp-E4h]
  int v118; // [rsp+2Ch] [rbp-E4h]
  int v119; // [rsp+2Ch] [rbp-E4h]
  unsigned int v120; // [rsp+2Ch] [rbp-E4h]
  unsigned int v121; // [rsp+2Ch] [rbp-E4h]
  int v122; // [rsp+30h] [rbp-E0h]
  __int64 v123; // [rsp+30h] [rbp-E0h]
  __int64 v124; // [rsp+30h] [rbp-E0h]
  __int64 v125; // [rsp+30h] [rbp-E0h]
  __int64 v126; // [rsp+30h] [rbp-E0h]
  __int64 v127; // [rsp+30h] [rbp-E0h]
  __int64 v128; // [rsp+30h] [rbp-E0h]
  unsigned int v129; // [rsp+38h] [rbp-D8h]
  unsigned int v130; // [rsp+38h] [rbp-D8h]
  __int64 v131; // [rsp+38h] [rbp-D8h]
  unsigned int v132; // [rsp+38h] [rbp-D8h]
  __int64 v133; // [rsp+38h] [rbp-D8h]
  unsigned int v134; // [rsp+38h] [rbp-D8h]
  __int64 v135; // [rsp+38h] [rbp-D8h]
  unsigned int v136; // [rsp+38h] [rbp-D8h]
  __int64 v137; // [rsp+38h] [rbp-D8h]
  __int64 v138; // [rsp+38h] [rbp-D8h]
  unsigned int v139; // [rsp+40h] [rbp-D0h]
  unsigned __int64 v140; // [rsp+40h] [rbp-D0h]
  unsigned __int64 v141; // [rsp+48h] [rbp-C8h]
  unsigned int v142; // [rsp+48h] [rbp-C8h]
  __int64 v143; // [rsp+48h] [rbp-C8h]
  unsigned int v144; // [rsp+48h] [rbp-C8h]
  __int64 v145; // [rsp+48h] [rbp-C8h]
  unsigned int v146; // [rsp+48h] [rbp-C8h]
  __int64 v147; // [rsp+48h] [rbp-C8h]
  unsigned int v148; // [rsp+48h] [rbp-C8h]
  __int64 v149; // [rsp+50h] [rbp-C0h]
  __int64 v150; // [rsp+58h] [rbp-B8h]
  __int64 v151; // [rsp+60h] [rbp-B0h]
  __int64 v152; // [rsp+68h] [rbp-A8h]
  unsigned int v154; // [rsp+70h] [rbp-A0h]
  __int64 *v156; // [rsp+80h] [rbp-90h]
  unsigned __int8 *v158; // [rsp+88h] [rbp-88h]
  __m128i v159; // [rsp+A0h] [rbp-70h] BYREF
  __m128i v160; // [rsp+B0h] [rbp-60h] BYREF
  unsigned __int64 v161; // [rsp+C0h] [rbp-50h] BYREF
  unsigned int v162; // [rsp+C8h] [rbp-48h]
  unsigned __int64 v163; // [rsp+D0h] [rbp-40h]
  unsigned int v164; // [rsp+D8h] [rbp-38h]

  v10 = sub_33D2320(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a1 + 40) + 48LL), a6, 0, 0, a6);
  if ( !v10 )
    return 0;
  v11 = *(_QWORD *)(v10 + 96);
  v12 = *(_DWORD *)(v11 + 32);
  v13 = v12 <= 0x40 ? *(_QWORD *)(v11 + 24) == 1 : v12 - 1 == (unsigned int)sub_C444A0(v11 + 24);
  if ( !v13 )
    return 0;
  v14 = **(_QWORD **)(a1 + 40);
  v152 = v14;
  if ( *(_DWORD *)(v14 + 24) != 56 )
    return 0;
  v16 = *(__int64 **)(v14 + 40);
  v17 = *v16;
  v18 = v16[5];
  v19 = v16[1];
  v20 = *((_DWORD *)v16 + 2);
  v151 = *v16;
  v150 = v16[6];
  v149 = v18;
  v139 = *((_DWORD *)v16 + 12);
  if ( *(_DWORD *)(*v16 + 24) == 56 )
  {
    v67 = *(_QWORD *)(v17 + 40);
    v123 = v18;
    v131 = v19;
    v142 = v20;
    v114 = *(_QWORD *)v67;
    v112 = *(_QWORD *)(v67 + 40);
    v120 = *(_DWORD *)(v67 + 8);
    v110 = *(_DWORD *)(v67 + 48);
    v68 = sub_33D2320(v112, *(_QWORD *)(v67 + 48), a6, 0, 0, v18);
    v69 = v142;
    v19 = v131;
    v70 = v123;
    if ( v68 )
    {
      v71 = *(_QWORD *)(v68 + 96);
      v72 = *(_DWORD *)(v71 + 32);
      if ( v72 <= 0x40 )
      {
        v21 = *(_QWORD *)(v71 + 24) == 1;
      }
      else
      {
        v73 = sub_C444A0(v71 + 24);
        v69 = v142;
        v19 = v131;
        v70 = v123;
        v21 = v72 - 1 == v73;
      }
      if ( v21 )
      {
        v116 = v17;
        v20 = v120;
        v151 = v114;
        goto LABEL_11;
      }
    }
    v124 = v19;
    v132 = v69;
    v143 = v70;
    v74 = sub_33D2320(v70, v150, a6, 0, 0, v70);
    v18 = v143;
    v20 = v132;
    v19 = v124;
    if ( v74 )
    {
      v75 = *(_QWORD *)(v74 + 96);
      v76 = *(_DWORD *)(v75 + 32);
      if ( v76 <= 0x40 )
      {
        v21 = *(_QWORD *)(v75 + 24) == 1;
      }
      else
      {
        v125 = v143;
        v133 = v19;
        v144 = v20;
        v77 = sub_C444A0(v75 + 24);
        v20 = v144;
        v19 = v133;
        v18 = v125;
        v21 = v76 - 1 == v77;
      }
      if ( v21 )
      {
        v116 = v17;
        v20 = v120;
        v139 = v110;
        v149 = v112;
        v151 = v114;
        goto LABEL_11;
      }
    }
  }
  v116 = 0;
  v21 = 0;
  if ( *(_DWORD *)(v18 + 24) != 56 )
    goto LABEL_11;
  v78 = *(_QWORD *)(v18 + 40);
  v126 = v19;
  v134 = v20;
  v145 = v18;
  v115 = *(_QWORD *)v78;
  v111 = *(_QWORD *)(v78 + 40);
  v121 = *(_DWORD *)(v78 + 8);
  v113 = *(_DWORD *)(v78 + 48);
  v79 = sub_33D2320(v111, *(_QWORD *)(v78 + 48), a6, 0, 0, v18);
  v80 = v145;
  v81 = v134;
  v19 = v126;
  if ( !v79 )
    goto LABEL_109;
  v82 = *(_QWORD *)(v79 + 96);
  v83 = *(_DWORD *)(v82 + 32);
  if ( v83 <= 0x40 )
  {
    v21 = *(_QWORD *)(v82 + 24) == 1;
  }
  else
  {
    v127 = v145;
    v135 = v19;
    v146 = v81;
    v84 = sub_C444A0(v82 + 24);
    v81 = v146;
    v19 = v135;
    v80 = v127;
    v21 = v83 - 1 == v84;
  }
  if ( v21 )
  {
    v139 = v81;
    v116 = v80;
    v20 = v121;
    v149 = v17;
    v151 = v115;
  }
  else
  {
LABEL_109:
    v136 = v81;
    v21 = 0;
    v147 = v19;
    v128 = v80;
    v85 = sub_33D2320(v17, v19, a6, 0, 0, v80);
    v19 = v147;
    v20 = v136;
    v116 = v85;
    if ( !v85 )
      goto LABEL_11;
    v86 = *(_QWORD *)(v85 + 96);
    v87 = v128;
    v88 = *(_DWORD *)(v86 + 32);
    if ( v88 <= 0x40 )
    {
      if ( *(_QWORD *)(v86 + 24) != 1 )
        goto LABEL_112;
    }
    else
    {
      v137 = v147;
      v148 = v20;
      v89 = sub_C444A0(v86 + 24);
      v20 = v148;
      v19 = v137;
      v87 = v128;
      if ( v89 != v88 - 1 )
      {
LABEL_112:
        v116 = 0;
        v21 = 0;
        goto LABEL_11;
      }
    }
    v116 = v87;
    v21 = v13;
    v20 = v121;
    v139 = v113;
    v149 = v111;
    v151 = v115;
  }
LABEL_11:
  v22 = *(_QWORD *)a3;
  v122 = *(_DWORD *)(a1 + 24);
  v141 = v20 | v19 & 0xFFFFFFFF00000000LL;
  v129 = sub_33D25A0(*(_QWORD *)a3, v151, v141, a6, a8);
  v140 = v139 | v150 & 0xFFFFFFFF00000000LL;
  v23 = sub_33D25A0(v22, v149, v140, a6, a8);
  v24 = v129;
  if ( v129 > v23 )
    v24 = v23;
  v130 = v24 - 1;
  sub_33D4EF0((__int64)&v161, v22, v151, v141, a6, a8);
  v25 = v162;
  if ( v162 > 0x40 )
  {
    v25 = sub_C44500((__int64)&v161);
    if ( v164 <= 0x40 || (v65 = v163) == 0 )
    {
LABEL_65:
      if ( v161 )
      {
        v118 = v25;
        j_j___libc_free_0_0(v161);
        v25 = v118;
      }
      goto LABEL_18;
    }
LABEL_80:
    v119 = v25;
    j_j___libc_free_0_0(v65);
    v25 = v119;
    if ( v162 <= 0x40 )
      goto LABEL_18;
    goto LABEL_65;
  }
  if ( v162 )
  {
    v25 = 64;
    if ( v161 << (64 - (unsigned __int8)v162) != -1 )
    {
      _BitScanReverse64(&v26, ~(v161 << (64 - (unsigned __int8)v162)));
      v25 = v26 ^ 0x3F;
    }
  }
  if ( v164 > 0x40 )
  {
    v65 = v163;
    if ( v163 )
      goto LABEL_80;
  }
LABEL_18:
  v117 = v25;
  sub_33D4EF0((__int64)&v161, v22, v149, v140, a6, a8);
  v27 = v162;
  if ( v162 > 0x40 )
  {
    v64 = sub_C44500((__int64)&v161);
    if ( v117 <= v64 )
      v64 = v117;
    v27 = v64;
    if ( v164 <= 0x40 || (v66 = v163) == 0 )
    {
LABEL_77:
      if ( v161 )
        j_j___libc_free_0_0(v161);
      goto LABEL_24;
    }
LABEL_83:
    j_j___libc_free_0_0(v66);
    if ( v162 <= 0x40 )
      goto LABEL_24;
    goto LABEL_77;
  }
  if ( v162 )
  {
    v28 = 64;
    if ( v161 << (64 - (unsigned __int8)v162) == -1 )
    {
      if ( v117 <= 0x40 )
        v28 = v117;
      v27 = v28;
    }
    else
    {
      _BitScanReverse64(&v29, ~(v161 << (64 - (unsigned __int8)v162)));
      v27 = v29 ^ 0x3F;
      if ( v27 > v117 )
        v27 = v117;
    }
  }
  if ( v164 > 0x40 )
  {
    v66 = v163;
    if ( v163 )
      goto LABEL_83;
  }
LABEL_24:
  if ( v122 == 191 )
  {
    v30 = 1;
    if ( v130 )
      v30 = v130;
    if ( v30 < v27 )
    {
LABEL_28:
      v13 = 0;
      v154 = !v21 ? 175 : 177;
      goto LABEL_29;
    }
    if ( !v130 )
      return 0;
    v27 = v130;
  }
  else
  {
    if ( v122 != 192 )
      BUG();
    if ( v130 < v27 )
      goto LABEL_28;
    v27 = v130;
    if ( !v130 || sub_986C60((__int64 *)a5, *(_DWORD *)(a5 + 8) - 1) )
      return 0;
  }
  if ( v21 )
    v13 = v21;
  v154 = !v21 ? 174 : 176;
LABEL_29:
  v31 = *(_QWORD *)(a1 + 48) + 16LL * a2;
  v32 = *(_WORD *)v31;
  v159.m128i_i64[1] = *(_QWORD *)(v31 + 8);
  v159.m128i_i16[0] = v32;
  v33 = sub_32844A0((unsigned __int16 *)&v159, v31) - v27;
  if ( v33 <= 7 || (_BitScanReverse(&v33, v33 - 1), v34 = v33 ^ 0x1F, v35 = 32 - v34, v34 == 29) )
  {
    v36 = 5;
    v37 = 0;
  }
  else
  {
    switch ( v35 )
    {
      case 4:
        v36 = 6;
        v37 = 0;
        break;
      case 5:
        v36 = 7;
        v37 = 0;
        break;
      case 6:
        v36 = 8;
        v37 = 0;
        break;
      case 7:
        v36 = 9;
        v37 = 0;
        break;
      default:
        v31 = (unsigned int)(1 << (32 - v34));
        v36 = sub_3007020(*(_QWORD **)(v22 + 64), v31);
        break;
    }
  }
  v160.m128i_i16[0] = v36;
  v160.m128i_i64[1] = v37;
  v38 = sub_32844A0((unsigned __int16 *)&v160, v31);
  if ( v38 > sub_32844A0((unsigned __int16 *)&v159, v31) )
    return 0;
  if ( v159.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v159.m128i_i16[0] - 17) > 0xD3u )
      goto LABEL_39;
    v92 = (unsigned __int16)(v159.m128i_i16[0] - 176) <= 0x34u;
    LODWORD(v90) = word_4456340[v159.m128i_u16[0] - 1];
    LOBYTE(v91) = v92;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v159) )
      goto LABEL_39;
    v90 = sub_3007240((__int64)&v159);
    v91 = HIDWORD(v90);
    v92 = BYTE4(v90);
  }
  v93 = *(__int64 **)(v22 + 64);
  v94 = v160.m128i_i32[0];
  LODWORD(v161) = v90;
  BYTE4(v161) = v91;
  v156 = v93;
  v138 = v160.m128i_i64[1];
  if ( v92 )
    v95 = sub_2D43AD0(v160.m128i_i16[0], v90);
  else
    v95 = sub_2D43050(v160.m128i_i16[0], v90);
  v98 = 0;
  if ( !v95 )
  {
    v95 = sub_3009450(v156, v94, v138, v161, v96, v97);
    v98 = v107;
  }
  v160.m128i_i16[0] = v95;
  v160.m128i_i64[1] = v98;
LABEL_39:
  if ( *(_BYTE *)(a3 + 8) && !sub_328D6E0(a4, v154, v160.m128i_u16[0]) )
  {
    if ( !*(_BYTE *)(a3 + 9) || sub_328D6E0(a4, v154, v159.m128i_u16[0]) )
    {
      v39 = *(_QWORD *)(v152 + 40);
      v40 = *(_QWORD *)(v39 + 40);
      v41 = *(_QWORD *)v39;
      v42 = *(unsigned int *)(v39 + 8);
      v43 = *(unsigned int *)(v39 + 48);
      if ( !(v13 ? sub_33DF4A0(v22, v41, v42, v40, v43) : (unsigned int)sub_33DD440(v22, v41, v42, v40, v43)) )
      {
        if ( !v116
          || ((v45 = *(_QWORD *)(v116 + 40),
               v46 = *(_QWORD *)(v45 + 40),
               v47 = *(_QWORD *)v45,
               v48 = *(unsigned int *)(v45 + 8),
               v49 = *(unsigned int *)(v45 + 48),
               !v13)
            ? (v50 = sub_33DD440(v22, v47, v48, v46, v49))
            : (v50 = sub_33DF4A0(v22, v47, v48, v46, v49)),
              !v50) )
        {
          a7 = _mm_loadu_si128(&v159);
          v160 = a7;
          goto LABEL_51;
        }
      }
    }
    return 0;
  }
LABEL_51:
  if ( !v21 )
  {
    if ( (v51 = 1, v160.m128i_i16[0] != 1)
      && (!v160.m128i_i16[0] || (v51 = v160.m128i_u16[0], !*(_QWORD *)(a4 + 8LL * v160.m128i_u16[0] + 112)))
      || *(_BYTE *)(v154 + a4 + 500 * v51 + 6414) )
    {
      v52 = *(_DWORD *)(v151 + 24);
      if ( v52 == 35 )
        return 0;
      if ( v52 == 11 )
        return 0;
      v53 = *(_DWORD *)(v149 + 24);
      if ( v53 == 35 || v53 == 11 )
        return 0;
    }
  }
  v54 = *(_QWORD *)(a1 + 80);
  v161 = v54;
  if ( v54 )
    sub_B96E90((__int64)&v161, v54, 1);
  v162 = *(_DWORD *)(a1 + 72);
  if ( v13 )
  {
    v55 = sub_33FB160(v22, v149, v140, (__int64)&v161, v160.m128i_i64[0], v160.m128i_i64[1], a7);
    v57 = v56;
    v58 = v55;
    *(_QWORD *)&v59 = sub_33FB160(v22, v151, v141, (__int64)&v161, v160.m128i_i64[0], v160.m128i_i64[1], a7);
    *((_QWORD *)&v108 + 1) = v57;
    *(_QWORD *)&v108 = v58;
    v61 = sub_3406EB0((_QWORD *)v22, v154, (__int64)&v161, v160.m128i_u32[0], v160.m128i_i64[1], v60, v59, v108);
    v63 = sub_33FB160(v22, (__int64)v61, v62, (__int64)&v161, v159.m128i_i64[0], v159.m128i_i64[1], a7);
  }
  else
  {
    v99 = sub_33FB310(v22, v149, v140, (__int64)&v161, v160.m128i_i64[0], v160.m128i_i64[1], a7);
    v101 = v100;
    v102 = v99;
    *(_QWORD *)&v103 = sub_33FB310(v22, v151, v141, (__int64)&v161, v160.m128i_i64[0], v160.m128i_i64[1], a7);
    *((_QWORD *)&v109 + 1) = v101;
    *(_QWORD *)&v109 = v102;
    v105 = sub_3406EB0((_QWORD *)v22, v154, (__int64)&v161, v160.m128i_u32[0], v160.m128i_i64[1], v104, v103, v109);
    v63 = sub_33FB310(v22, (__int64)v105, v106, (__int64)&v161, v159.m128i_i64[0], v159.m128i_i64[1], a7);
  }
  result = v63;
  if ( v161 )
  {
    v158 = v63;
    sub_B91220((__int64)&v161, v161);
    return v158;
  }
  return result;
}

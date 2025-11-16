// Function: sub_1D6E200
// Address: 0x1d6e200
//
__int64 __fastcall sub_1D6E200(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 *v10; // rax
  __int64 v11; // r15
  __int64 v12; // rdx
  __int64 *v13; // rdi
  __int64 *v14; // r13
  __int64 *v15; // rbx
  __int64 v16; // rax
  __int64 *v17; // r12
  unsigned __int64 v18; // r14
  __int64 *v19; // rsi
  __int64 *v20; // r14
  __int64 *v21; // rdi
  __int64 v22; // rax
  __int64 *v23; // rcx
  __int64 v24; // rdx
  __int64 *v25; // rsi
  __int64 v26; // rdi
  __int64 *v27; // rax
  __int64 *v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // r8
  __int64 v31; // rsi
  __int64 *v32; // rdx
  __int64 v33; // rdi
  __int64 v34; // r12
  unsigned __int64 v35; // r14
  _QWORD *v36; // rax
  unsigned __int8 *v37; // rsi
  __int64 v38; // rax
  __int64 v39; // rax
  double v40; // xmm4_8
  double v41; // xmm5_8
  __int64 v42; // r9
  __int64 v43; // rsi
  int v44; // eax
  int v45; // esi
  __int64 v46; // rcx
  unsigned int v47; // edx
  __int64 *v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rcx
  __int64 v51; // rdx
  __int64 *v52; // rax
  __int64 v53; // rcx
  __int64 v54; // rax
  __int64 v55; // rdi
  __int64 v56; // rdx
  bool v57; // zf
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  unsigned __int8 v61; // al
  __int64 v62; // r14
  __int64 v63; // rdx
  __int64 v64; // rax
  unsigned __int8 *v65; // rsi
  __int64 v66; // r14
  __int64 v67; // r9
  int v68; // r8d
  int v69; // r9d
  unsigned __int64 v70; // rdx
  unsigned __int64 *v71; // rsi
  unsigned __int64 *v72; // rcx
  unsigned __int64 *v73; // rax
  _QWORD *v74; // r8
  unsigned __int64 v75; // rdx
  _QWORD *v76; // rax
  _BOOL4 v77; // r9d
  __int64 v78; // rax
  int v79; // eax
  int v80; // r8d
  __int64 v81; // rdi
  __int64 v82; // r9
  __int64 v83; // rax
  __int64 v84; // rcx
  __int64 v85; // rsi
  __int64 v86; // rdx
  unsigned __int8 *v87; // rsi
  unsigned __int64 *v88; // r14
  _QWORD *v89; // r12
  __int64 v90; // r13
  unsigned __int64 v91; // rdi
  _QWORD *i; // r15
  unsigned __int64 v93; // rcx
  _QWORD *v94; // rax
  _BOOL4 v95; // r12d
  __int64 v96; // rax
  unsigned int v97; // eax
  _QWORD *v98; // rax
  __int64 v99; // rax
  __int64 v100; // r9
  __int64 v101; // rsi
  __int64 v102; // rax
  __int64 v103; // rsi
  __int64 v104; // rdx
  unsigned __int8 *v105; // rsi
  __int64 *v106; // rax
  __int64 v107; // rsi
  __int64 *v108; // [rsp+8h] [rbp-188h]
  __int64 v109; // [rsp+10h] [rbp-180h]
  _QWORD *v110; // [rsp+18h] [rbp-178h]
  __int64 *v111; // [rsp+20h] [rbp-170h]
  _BOOL4 v112; // [rsp+28h] [rbp-168h]
  __int64 *v113; // [rsp+28h] [rbp-168h]
  _QWORD *v114; // [rsp+28h] [rbp-168h]
  __int64 v115; // [rsp+30h] [rbp-160h]
  _QWORD *v116; // [rsp+30h] [rbp-160h]
  unsigned __int64 v117; // [rsp+30h] [rbp-160h]
  _QWORD *v118; // [rsp+30h] [rbp-160h]
  __int64 v119; // [rsp+30h] [rbp-160h]
  __int64 v120; // [rsp+30h] [rbp-160h]
  __int64 v121; // [rsp+30h] [rbp-160h]
  __int64 v122; // [rsp+38h] [rbp-158h]
  __int64 v123; // [rsp+40h] [rbp-150h]
  __int64 v124; // [rsp+48h] [rbp-148h]
  __int64 v125; // [rsp+50h] [rbp-140h]
  unsigned __int8 v126; // [rsp+58h] [rbp-138h]
  __int64 v127; // [rsp+58h] [rbp-138h]
  __int64 *v128; // [rsp+58h] [rbp-138h]
  __int64 v129; // [rsp+60h] [rbp-130h]
  __int64 v130; // [rsp+60h] [rbp-130h]
  __int64 v131; // [rsp+60h] [rbp-130h]
  __int64 v132; // [rsp+60h] [rbp-130h]
  __int64 v133; // [rsp+68h] [rbp-128h]
  unsigned __int8 *v134; // [rsp+78h] [rbp-118h] BYREF
  __int64 v135[2]; // [rsp+80h] [rbp-110h] BYREF
  __int16 v136; // [rsp+90h] [rbp-100h]
  unsigned __int8 *v137[2]; // [rsp+A0h] [rbp-F0h] BYREF
  __int16 v138; // [rsp+B0h] [rbp-E0h]
  unsigned __int8 *v139; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v140; // [rsp+C8h] [rbp-C8h]
  __int64 *v141; // [rsp+D0h] [rbp-C0h]
  _QWORD *v142; // [rsp+D8h] [rbp-B8h]
  __int64 v143; // [rsp+E0h] [rbp-B0h]
  int v144; // [rsp+E8h] [rbp-A8h]
  __int64 v145; // [rsp+F0h] [rbp-A0h]
  __int64 v146; // [rsp+F8h] [rbp-98h]
  unsigned __int8 *v147; // [rsp+110h] [rbp-80h] BYREF
  __int64 v148; // [rsp+118h] [rbp-78h]
  __int64 *v149; // [rsp+120h] [rbp-70h]
  __int64 v150; // [rsp+128h] [rbp-68h]
  __int64 v151; // [rsp+130h] [rbp-60h]
  int v152; // [rsp+138h] [rbp-58h]
  __int64 v153; // [rsp+140h] [rbp-50h]
  __int64 v154; // [rsp+148h] [rbp-48h]

  if ( !*(_DWORD *)(a1 + 736) )
    return 0;
  v10 = *(__int64 **)(a1 + 728);
  v11 = a1;
  v12 = 67LL * *(unsigned int *)(a1 + 744);
  v13 = &v10[v12];
  v111 = &v10[v12];
  if ( v10 == &v10[v12] )
    return 0;
  while ( 1 )
  {
    v14 = v10;
    if ( *v10 != -16 && *v10 != -8 )
      break;
    v10 += 67;
    if ( v13 == v10 )
      return 0;
  }
  v122 = *v10;
  if ( v111 == v10 )
    return 0;
  v126 = 0;
  v110 = (_QWORD *)(v11 + 792);
  while ( 2 )
  {
    v15 = (__int64 *)v14[1];
    v16 = 16LL * *((unsigned int *)v14 + 4);
    v17 = (__int64 *)((char *)v15 + v16);
    v18 = v16;
    if ( (__int64 *)((char *)v15 + v16) == v15 )
      goto LABEL_116;
    v19 = (__int64 *)((char *)v15 + v16);
    _BitScanReverse64((unsigned __int64 *)&v16, v16 >> 4);
    sub_1D6CC00((__int64 *)v14[1], v19, 2LL * (int)(63 - (v16 ^ 0x3F)), v11);
    if ( v18 <= 0x100 )
    {
      sub_1D6BBD0(v15, v17, v11);
    }
    else
    {
      v20 = v15 + 32;
      sub_1D6BBD0(v15, v15 + 32, v11);
      if ( v17 != v15 + 32 )
      {
        do
        {
          v21 = v20;
          v20 += 2;
          sub_1D6B800(v21, v11);
        }
        while ( v17 != v20 );
      }
    }
    v15 = (__int64 *)v14[1];
    v22 = 2LL * *((unsigned int *)v14 + 4);
    v23 = &v15[v22];
    v24 = v22 * 8;
    if ( &v15[v22] == v15 )
    {
LABEL_116:
      LODWORD(v24) = 0;
      v107 = -16;
      v22 = 0;
      goto LABEL_32;
    }
    v25 = v15 + 2;
    if ( v23 == v15 + 2 )
    {
      v107 = v22 * 8 - 16;
      LODWORD(v24) = (v22 * 8) >> 4;
      goto LABEL_32;
    }
    while ( 1 )
    {
      v26 = *(v25 - 2);
      v27 = v25 - 2;
      if ( v26 == *v25 && *(v25 - 1) == v25[1] )
        break;
      v25 += 2;
      if ( v23 == v25 )
      {
        v22 = 2LL * *((unsigned int *)v14 + 4);
        v107 = v24 - 16;
        LODWORD(v24) = v24 >> 4;
        goto LABEL_32;
      }
    }
    if ( v23 == v27 )
    {
      v24 = ((char *)v23 - (char *)v15) >> 4;
      v22 = 2LL * (unsigned int)v24;
      v107 = v22 * 8 - 16;
      goto LABEL_32;
    }
    v28 = v25 + 2;
    if ( v23 == v25 + 2 )
    {
      v106 = v25;
LABEL_175:
      v24 = ((char *)v106 - (char *)v15) >> 4;
      v22 = 2LL * (unsigned int)v24;
      v107 = v22 * 8 - 16;
      goto LABEL_32;
    }
    while ( v26 != *v28 || v27[1] != v28[1] )
    {
      v27[2] = *v28;
      v29 = v28[1];
      v28 += 2;
      v27 += 2;
      v27[1] = v29;
      if ( v23 == v28 )
        goto LABEL_28;
LABEL_24:
      v26 = *v27;
    }
    v28 += 2;
    if ( v23 != v28 )
      goto LABEL_24;
LABEL_28:
    v15 = (__int64 *)v14[1];
    v106 = v27 + 2;
    v30 = (char *)&v15[2 * *((unsigned int *)v14 + 4)] - (char *)v23;
    v31 = v30 >> 4;
    if ( v30 <= 0 )
      goto LABEL_175;
    v32 = v106;
    do
    {
      v33 = *v23;
      v32 += 2;
      v23 += 2;
      *(v32 - 2) = v33;
      *(v32 - 1) = *(v23 - 1);
      --v31;
    }
    while ( v31 );
    v15 = (__int64 *)v14[1];
    LODWORD(v24) = ((char *)v106 + v30 - (char *)v15) >> 4;
    v22 = 2LL * (unsigned int)v24;
    v107 = v22 * 8 - 16;
LABEL_32:
    *((_DWORD *)v14 + 4) = v24;
    v129 = v15[1];
    if ( v129 == *(__int64 *)((char *)v15 + v107 + 8) )
      goto LABEL_95;
    v34 = *v15;
    v133 = v15[1];
    v35 = 0;
    v123 = *v15;
    if ( !(v22 * 8) )
      goto LABEL_95;
    while ( 2 )
    {
      v36 = (_QWORD *)sub_16498A0(v34);
      v145 = 0;
      v146 = 0;
      v37 = *(unsigned __int8 **)(v34 + 48);
      v142 = v36;
      v144 = 0;
      v38 = *(_QWORD *)(v34 + 40);
      v139 = 0;
      v140 = v38;
      v143 = 0;
      v141 = (__int64 *)(v34 + 24);
      v147 = v37;
      if ( v37 )
      {
        sub_1623A60((__int64)&v147, (__int64)v37, 2);
        if ( v139 )
          sub_161E7C0((__int64)&v139, (__int64)v139);
        v139 = v147;
        if ( v147 )
          sub_1623210((__int64)&v147, v147, (__int64)&v139);
      }
      v125 = sub_15A9650(*(_QWORD *)(v11 + 904), *(_QWORD *)v34);
      v39 = *(_QWORD *)v34;
      if ( *(_BYTE *)(*(_QWORD *)v34 + 8LL) == 16 )
        v39 = **(_QWORD **)(v39 + 16);
      v127 = sub_16471D0(v142, *(_DWORD *)(v39 + 8) >> 8);
      v124 = sub_1643330(v142);
      if ( v35 )
        goto LABEL_42;
      v61 = *(_BYTE *)(v122 + 16);
      if ( v61 <= 0x17u )
      {
        v62 = *(_QWORD *)(sub_15F2060(v123) + 80);
        if ( v62 )
          v62 -= 24;
        goto LABEL_105;
      }
      v62 = *(_QWORD *)(v122 + 40);
      if ( v61 == 77 )
      {
LABEL_105:
        v63 = sub_157EE30(v62);
        goto LABEL_75;
      }
      if ( v61 == 29 )
      {
        v62 = sub_1AA91E0(*(_QWORD **)(v122 + 40), *(_QWORD **)(v122 - 48), 0, 0);
        v63 = sub_157EE30(v62);
      }
      else
      {
        v63 = *(_QWORD *)(v122 + 32);
      }
LABEL_75:
      v115 = v63;
      v64 = sub_157E9C0(v62);
      v148 = v62;
      v147 = 0;
      v150 = v64;
      v151 = 0;
      v152 = 0;
      v153 = 0;
      v154 = 0;
      v149 = (__int64 *)v115;
      if ( v115 != v62 + 40 )
      {
        if ( !v115 )
          BUG();
        v65 = *(unsigned __int8 **)(v115 + 24);
        v137[0] = v65;
        if ( v65 )
        {
          sub_1623A60((__int64)v137, (__int64)v65, 2);
          if ( v147 )
            sub_161E7C0((__int64)&v147, (__int64)v147);
          v147 = v137[0];
          if ( v137[0] )
            sub_1623210((__int64)v137, v137[0], (__int64)&v147);
        }
      }
      v66 = sub_15A0680(v125, v133, 0);
      v67 = v122;
      if ( v127 != *(_QWORD *)v122 )
      {
        v136 = 257;
        if ( v127 != *(_QWORD *)v122 )
        {
          if ( *(_BYTE *)(v122 + 16) > 0x10u )
          {
            v138 = 257;
            v99 = sub_15FDFF0(v122, v127, (__int64)v137, 0);
            v100 = v99;
            if ( v148 )
            {
              v113 = v149;
              v119 = v99;
              sub_157E9D0(v148 + 40, v99);
              v100 = v119;
              v101 = *v113;
              v102 = *(_QWORD *)(v119 + 24);
              *(_QWORD *)(v119 + 32) = v113;
              v101 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v119 + 24) = v101 | v102 & 7;
              *(_QWORD *)(v101 + 8) = v119 + 24;
              *v113 = *v113 & 7 | (v119 + 24);
            }
            v120 = v100;
            sub_164B780(v100, v135);
            v67 = v120;
            if ( v147 )
            {
              v134 = v147;
              sub_1623A60((__int64)&v134, (__int64)v147, 2);
              v67 = v120;
              v103 = *(_QWORD *)(v120 + 48);
              v104 = v120 + 48;
              if ( v103 )
              {
                sub_161E7C0(v120 + 48, v103);
                v67 = v120;
                v104 = v120 + 48;
              }
              v105 = v134;
              *(_QWORD *)(v67 + 48) = v134;
              if ( v105 )
              {
                v121 = v67;
                sub_1623210((__int64)&v134, v105, v104);
                v67 = v121;
              }
            }
          }
          else
          {
            v67 = sub_15A4A70((__int64 ***)v122, v127);
          }
        }
      }
      v137[0] = "splitgep";
      v138 = 259;
      v35 = sub_12815B0((__int64 *)&v147, v124, (_BYTE *)v67, v66, (__int64)v137);
      if ( *(_QWORD *)(v11 + 824) )
      {
        v74 = *(_QWORD **)(v11 + 800);
        if ( v74 )
        {
          while ( 1 )
          {
            v75 = v74[4];
            v76 = (_QWORD *)v74[3];
            if ( v35 < v75 )
              v76 = (_QWORD *)v74[2];
            if ( !v76 )
              break;
            v74 = v76;
          }
          if ( v35 >= v75 )
            goto LABEL_113;
          if ( v74 == *(_QWORD **)(v11 + 808) )
            goto LABEL_114;
        }
        else
        {
          v74 = v110;
          if ( v110 == *(_QWORD **)(v11 + 808) )
            goto LABEL_180;
        }
        v114 = v74;
        if ( v35 <= *(_QWORD *)(sub_220EF80(v74) + 32) )
          goto LABEL_92;
        v74 = v114;
        if ( !v114 )
          goto LABEL_92;
        goto LABEL_114;
      }
      v70 = *(unsigned int *)(v11 + 760);
      v71 = *(unsigned __int64 **)(v11 + 752);
      v72 = &v71[v70];
      if ( v71 != v72 )
      {
        v73 = *(unsigned __int64 **)(v11 + 752);
        while ( v35 != *v73 )
        {
          if ( v72 == ++v73 )
            goto LABEL_131;
        }
        if ( v72 != v73 )
          goto LABEL_92;
      }
LABEL_131:
      if ( v70 <= 1 )
      {
        if ( *(_DWORD *)(v11 + 760) >= *(_DWORD *)(v11 + 764) )
        {
          sub_16CD150(v11 + 752, (const void *)(v11 + 768), 0, 8, v68, v69);
          v72 = (unsigned __int64 *)(*(_QWORD *)(v11 + 752) + 8LL * *(unsigned int *)(v11 + 760));
        }
        *v72 = v35;
        ++*(_DWORD *)(v11 + 760);
        goto LABEL_92;
      }
      v109 = v34;
      v117 = v35;
      v88 = &v71[v70 - 1];
      v89 = *(_QWORD **)(v11 + 800);
      v108 = v14;
      v90 = v11;
      if ( v89 )
      {
LABEL_133:
        v91 = *v88;
        for ( i = v89; ; i = v94 )
        {
          v93 = i[4];
          v94 = (_QWORD *)i[3];
          if ( v91 < v93 )
            v94 = (_QWORD *)i[2];
          if ( !v94 )
            break;
        }
        if ( v91 < v93 )
        {
          if ( i != *(_QWORD **)(v90 + 808) )
            goto LABEL_148;
        }
        else if ( v91 <= v93 )
        {
          goto LABEL_143;
        }
LABEL_140:
        v95 = 1;
        if ( v110 != i )
          v95 = *v88 < i[4];
LABEL_142:
        v96 = sub_22077B0(40);
        *(_QWORD *)(v96 + 32) = *v88;
        sub_220F040(v95, v96, i, v110);
        ++*(_QWORD *)(v90 + 824);
        v89 = *(_QWORD **)(v90 + 800);
        goto LABEL_143;
      }
      while ( 1 )
      {
        i = v110;
        if ( v110 == *(_QWORD **)(v90 + 808) )
        {
          v95 = 1;
          goto LABEL_142;
        }
LABEL_148:
        if ( *(_QWORD *)(sub_220EF80(i) + 32) < *v88 )
          goto LABEL_140;
LABEL_143:
        v97 = *(_DWORD *)(v90 + 760) - 1;
        *(_DWORD *)(v90 + 760) = v97;
        if ( !v97 )
          break;
        v88 = (unsigned __int64 *)(*(_QWORD *)(v90 + 752) + 8LL * v97 - 8);
        if ( v89 )
          goto LABEL_133;
      }
      v74 = v89;
      v11 = v90;
      v35 = v117;
      v34 = v109;
      v14 = v108;
      if ( v74 )
      {
        while ( 1 )
        {
          v75 = v74[4];
          v98 = (_QWORD *)v74[3];
          if ( v117 < v75 )
            v98 = (_QWORD *)v74[2];
          if ( !v98 )
            break;
          v74 = v98;
        }
        if ( v117 < v75 )
        {
          if ( *(_QWORD **)(v11 + 808) != v74 )
          {
LABEL_160:
            v118 = v74;
            if ( v35 <= *(_QWORD *)(sub_220EF80(v74) + 32) )
              goto LABEL_92;
            v74 = v118;
            if ( !v118 )
              goto LABEL_92;
            v77 = 1;
            if ( v110 == v118 )
              goto LABEL_115;
LABEL_163:
            v77 = v35 < v74[4];
            goto LABEL_115;
          }
LABEL_114:
          v77 = 1;
          if ( v110 == v74 )
            goto LABEL_115;
          goto LABEL_163;
        }
LABEL_113:
        if ( v35 <= v75 )
          goto LABEL_92;
        goto LABEL_114;
      }
      v74 = v110;
      if ( v110 != *(_QWORD **)(v11 + 808) )
        goto LABEL_160;
LABEL_180:
      v74 = v110;
      v77 = 1;
LABEL_115:
      v112 = v77;
      v116 = v74;
      v78 = sub_22077B0(40);
      *(_QWORD *)(v78 + 32) = v35;
      sub_220F040(v112, v78, v116, v110);
      ++*(_QWORD *)(v11 + 824);
LABEL_92:
      if ( v147 )
        sub_161E7C0((__int64)&v147, (__int64)v147);
LABEL_42:
      if ( v133 == v129 )
      {
        v42 = v35;
        if ( v127 != *(_QWORD *)v34 )
        {
          v138 = 257;
          v43 = *(_QWORD *)v34;
          if ( *(_QWORD *)v34 != *(_QWORD *)v35 )
          {
            if ( *(_BYTE *)(v35 + 16) <= 0x10u )
            {
              v42 = sub_15A4A70((__int64 ***)v35, v43);
              goto LABEL_47;
            }
            LOWORD(v149) = 257;
            v81 = v35;
LABEL_123:
            v82 = sub_15FDFF0(v81, v43, (__int64)&v147, 0);
            if ( v140 )
            {
              v130 = v82;
              v128 = v141;
              sub_157E9D0(v140 + 40, v82);
              v82 = v130;
              v83 = *(_QWORD *)(v130 + 24);
              v84 = *v128;
              *(_QWORD *)(v130 + 32) = v128;
              v84 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v130 + 24) = v84 | v83 & 7;
              *(_QWORD *)(v84 + 8) = v130 + 24;
              *v128 = *v128 & 7 | (v130 + 24);
            }
            v131 = v82;
            sub_164B780(v82, (__int64 *)v137);
            v42 = v131;
            if ( v139 )
            {
              v135[0] = (__int64)v139;
              sub_1623A60((__int64)v135, (__int64)v139, 2);
              v42 = v131;
              v85 = *(_QWORD *)(v131 + 48);
              v86 = v131 + 48;
              if ( v85 )
              {
                sub_161E7C0(v131 + 48, v85);
                v42 = v131;
                v86 = v131 + 48;
              }
              v87 = (unsigned __int8 *)v135[0];
              *(_QWORD *)(v42 + 48) = v135[0];
              if ( v87 )
              {
                v132 = v42;
                sub_1623210((__int64)v135, v87, v86);
                v42 = v132;
              }
            }
          }
        }
      }
      else
      {
        v60 = sub_15A0680(v125, v129 - v133, 0);
        LOWORD(v149) = 257;
        v42 = sub_12815B0((__int64 *)&v139, v124, (_BYTE *)v35, v60, (__int64)&v147);
        if ( v127 != *(_QWORD *)v34 )
        {
          v138 = 257;
          v43 = *(_QWORD *)v34;
          if ( *(_QWORD *)v34 != *(_QWORD *)v42 )
          {
            if ( *(_BYTE *)(v42 + 16) > 0x10u )
            {
              v81 = v42;
              LOWORD(v149) = 257;
              goto LABEL_123;
            }
            v42 = sub_15A4A70((__int64 ***)v42, v43);
          }
        }
      }
LABEL_47:
      sub_164D160(v34, v42, a2, a3, a4, a5, v40, v41, a8, a9);
      v44 = *(_DWORD *)(v11 + 856);
      if ( v44 )
      {
        v45 = v44 - 1;
        v46 = *(_QWORD *)(v11 + 840);
        v47 = (v44 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
        v48 = (__int64 *)(v46 + 16LL * v47);
        v49 = *v48;
        if ( *v48 == v34 )
        {
LABEL_49:
          *v48 = -16;
          --*(_DWORD *)(v11 + 848);
          ++*(_DWORD *)(v11 + 852);
        }
        else
        {
          v79 = 1;
          while ( v49 != -8 )
          {
            v80 = v79 + 1;
            v47 = v45 & (v79 + v47);
            v48 = (__int64 *)(v46 + 16LL * v47);
            v49 = *v48;
            if ( v34 == *v48 )
              goto LABEL_49;
            v79 = v80;
          }
        }
      }
      v50 = *((unsigned int *)v14 + 4);
      v51 = (v14[1] + 16 * v50 - (__int64)(v15 + 2)) >> 4;
      if ( v14[1] + 16 * v50 - (__int64)(v15 + 2) > 0 )
      {
        v52 = v15;
        do
        {
          v53 = v52[2];
          v52 += 2;
          *(v52 - 2) = v53;
          *(v52 - 1) = v52[1];
          --v51;
        }
        while ( v51 );
        LODWORD(v50) = *((_DWORD *)v14 + 4);
      }
      *((_DWORD *)v14 + 4) = v50 - 1;
      sub_15F20C0((_QWORD *)v34);
      if ( v139 )
        sub_161E7C0((__int64)&v139, (__int64)v139);
      if ( v15 != (__int64 *)(v14[1] + 16LL * *((unsigned int *)v14 + 4)) )
      {
        v54 = v15[1];
        v34 = *v15;
        v129 = v54;
        if ( v133 != v54 )
        {
          v147 = 0;
          v55 = *(_QWORD *)(v11 + 176);
          LOBYTE(v149) = 0;
          v150 = 0;
          v148 = v54 - v133;
          v56 = **(_QWORD **)(v34 - 24LL * (*(_DWORD *)(v34 + 20) & 0xFFFFFFF));
          if ( *(_BYTE *)(v56 + 8) == 16 )
            v56 = **(_QWORD **)(v56 + 16);
          v57 = (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, unsigned __int8 **, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v55 + 736LL))(
                  v55,
                  *(_QWORD *)(v11 + 904),
                  &v147,
                  *(_QWORD *)(v34 + 64),
                  *(_DWORD *)(v56 + 8) >> 8,
                  0) == 0;
          v58 = v133;
          if ( v57 )
            v58 = v129;
          v133 = v58;
          v59 = v123;
          if ( v57 )
            v59 = v34;
          v123 = v59;
          if ( v57 )
            v35 = 0;
        }
        continue;
      }
      break;
    }
    v126 = 1;
LABEL_95:
    v14 += 67;
    if ( v14 != v111 )
    {
      while ( *v14 == -8 || *v14 == -16 )
      {
        v14 += 67;
        if ( v111 == v14 )
          return v126;
      }
      if ( v111 != v14 )
      {
        v122 = *v14;
        continue;
      }
    }
    return v126;
  }
}

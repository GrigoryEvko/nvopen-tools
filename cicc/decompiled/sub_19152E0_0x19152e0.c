// Function: sub_19152E0
// Address: 0x19152e0
//
__int64 __fastcall sub_19152E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 *v12; // r8
  __int64 *v13; // r9
  __int64 *v15; // r15
  __int64 v16; // r12
  __int64 v17; // rsi
  __int64 *v18; // rdi
  __int64 *v19; // rax
  __int64 *v20; // rcx
  __int64 *m128i_i64; // r12
  __int64 v22; // r15
  unsigned __int64 v23; // rdi
  __int64 *v24; // rax
  __int64 *v25; // r13
  unsigned __int64 v27; // rax
  int v28; // eax
  int v29; // edx
  int v30; // esi
  __int64 v31; // rdi
  unsigned int v32; // edx
  __int64 v33; // rcx
  __int64 *v34; // rdx
  __int64 *v35; // r12
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rdi
  unsigned int v39; // esi
  __int64 *v40; // r14
  unsigned int v41; // edx
  __int64 v42; // rax
  __int64 v43; // r8
  int v44; // edx
  __int64 v45; // rdx
  __int64 *v46; // r12
  __int64 v47; // r13
  unsigned int v48; // ecx
  __int64 v49; // rax
  __int64 v50; // rdi
  const char *v51; // rdx
  int v52; // esi
  int v53; // ecx
  __int64 v54; // r13
  _QWORD *v55; // rax
  __int64 v56; // r12
  __int64 v57; // r14
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rax
  unsigned __int64 v60; // rax
  int v61; // r8d
  int v62; // r9d
  __int64 v63; // rax
  __int64 *v64; // rdi
  __int64 *v65; // r14
  __int64 *v66; // r13
  __int64 v67; // rsi
  __int64 v68; // rax
  __int64 *v69; // r13
  __int64 v70; // rcx
  __int64 v71; // rdx
  __int64 v72; // rax
  __int64 v73; // rax
  unsigned int j; // eax
  int v75; // r8d
  int v76; // r9d
  __int64 v77; // rax
  __int64 v78; // r10
  int v79; // r11d
  __int64 v80; // r10
  int v81; // r11d
  __m128i **v82; // r12
  __m128i **v83; // r14
  __m128i *v84; // r15
  __int64 v85; // rsi
  unsigned __int8 *v86; // rsi
  __m128i *v87; // r8
  int v88; // r8d
  int v89; // r9d
  __int64 v90; // rdx
  __int64 v91; // rdx
  __int64 v92; // rdx
  __int64 v93; // rax
  __m128 *v94; // rax
  __int64 v95; // r12
  unsigned int v96; // eax
  __int64 v97; // rdx
  char v98; // cl
  __int64 v99; // r12
  __int64 v100; // rsi
  __int64 v101; // rsi
  unsigned __int8 *v102; // rsi
  __int64 *v103; // r12
  double v104; // xmm4_8
  double v105; // xmm5_8
  unsigned __int8 v106; // al
  __m128i *v107; // r13
  __int64 v108; // rsi
  char v109; // al
  int v110; // r8d
  int v111; // r9d
  __int64 v112; // rax
  __int64 *v113; // r12
  __int64 v114; // rax
  __m128i v115; // xmm1
  __m128i v116; // xmm2
  __int64 v117; // rax
  __int64 v118; // rax
  unsigned __int8 v119; // [rsp+4h] [rbp-58Ch]
  unsigned int v120; // [rsp+8h] [rbp-588h]
  char v121; // [rsp+Ch] [rbp-584h]
  unsigned __int64 v122; // [rsp+10h] [rbp-580h]
  int v123; // [rsp+20h] [rbp-570h]
  __m128i *v124; // [rsp+20h] [rbp-570h]
  __int64 *v125; // [rsp+28h] [rbp-568h]
  __int64 v126; // [rsp+30h] [rbp-560h]
  __int64 v127; // [rsp+40h] [rbp-550h]
  __int64 *v128; // [rsp+48h] [rbp-548h]
  __int64 v130; // [rsp+50h] [rbp-540h]
  __int64 *v131; // [rsp+50h] [rbp-540h]
  const char *v134; // [rsp+68h] [rbp-528h]
  __int64 *v135; // [rsp+68h] [rbp-528h]
  __int64 v136; // [rsp+70h] [rbp-520h] BYREF
  __int64 v137; // [rsp+78h] [rbp-518h]
  __int64 v138; // [rsp+80h] [rbp-510h]
  unsigned int v139; // [rsp+88h] [rbp-508h]
  __int64 *v140; // [rsp+90h] [rbp-500h] BYREF
  __int64 v141; // [rsp+98h] [rbp-4F8h]
  _BYTE v142[32]; // [rsp+A0h] [rbp-4F0h] BYREF
  __int64 v143; // [rsp+C0h] [rbp-4D0h] BYREF
  __int64 v144; // [rsp+C8h] [rbp-4C8h]
  __int64 v145; // [rsp+D0h] [rbp-4C0h]
  int v146; // [rsp+D8h] [rbp-4B8h]
  __int64 *v147; // [rsp+E0h] [rbp-4B0h]
  __int64 *v148; // [rsp+E8h] [rbp-4A8h]
  __int64 v149; // [rsp+F0h] [rbp-4A0h]
  __int64 v150; // [rsp+100h] [rbp-490h] BYREF
  __int64 *v151; // [rsp+108h] [rbp-488h]
  __int64 *v152; // [rsp+110h] [rbp-480h]
  __int64 v153; // [rsp+118h] [rbp-478h]
  int i; // [rsp+120h] [rbp-470h]
  _BYTE v155[40]; // [rsp+128h] [rbp-468h] BYREF
  __m128i **v156; // [rsp+150h] [rbp-440h] BYREF
  __int64 v157; // [rsp+158h] [rbp-438h]
  _BYTE v158[64]; // [rsp+160h] [rbp-430h] BYREF
  const char *v159; // [rsp+1A0h] [rbp-3F0h] BYREF
  __int64 v160; // [rsp+1A8h] [rbp-3E8h]
  unsigned __int64 v161; // [rsp+1B0h] [rbp-3E0h]
  __m128i v162; // [rsp+1B8h] [rbp-3D8h]
  __int64 v163; // [rsp+1C8h] [rbp-3C8h]
  __int64 v164; // [rsp+1D0h] [rbp-3C0h]
  __m128i v165; // [rsp+1D8h] [rbp-3B8h]
  __int64 v166; // [rsp+1E8h] [rbp-3A8h]
  char v167; // [rsp+1F0h] [rbp-3A0h]
  _QWORD v168[2]; // [rsp+1F8h] [rbp-398h] BYREF
  _BYTE v169[356]; // [rsp+208h] [rbp-388h] BYREF
  int v170; // [rsp+36Ch] [rbp-224h]
  __int64 v171; // [rsp+370h] [rbp-220h]
  __m128i v172; // [rsp+380h] [rbp-210h] BYREF
  unsigned __int64 v173; // [rsp+390h] [rbp-200h]
  __m128i v174; // [rsp+398h] [rbp-1F8h] BYREF
  __int64 v175; // [rsp+3A8h] [rbp-1E8h]
  __int64 v176; // [rsp+3B0h] [rbp-1E0h] BYREF
  __m128i v177; // [rsp+3B8h] [rbp-1D8h] BYREF
  __int64 v178; // [rsp+3C8h] [rbp-1C8h]
  char v179; // [rsp+3D0h] [rbp-1C0h]
  char v180[8]; // [rsp+3D8h] [rbp-1B8h] BYREF
  int v181; // [rsp+3E0h] [rbp-1B0h]
  char v182; // [rsp+548h] [rbp-48h]
  int v183; // [rsp+54Ch] [rbp-44h]
  __int64 v184; // [rsp+550h] [rbp-40h]

  v12 = (__int64 *)v155;
  v13 = (__int64 *)v155;
  v15 = *(__int64 **)a4;
  v16 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
  v150 = 0;
  v151 = (__int64 *)v155;
  v152 = (__int64 *)v155;
  v153 = 4;
  for ( i = 0; (__int64 *)v16 != v15; ++v15 )
  {
LABEL_5:
    v17 = *v15;
    if ( v12 != v13 )
      goto LABEL_3;
    v18 = &v12[HIDWORD(v153)];
    if ( v18 != v12 )
    {
      v19 = v12;
      v20 = 0;
      while ( v17 != *v19 )
      {
        if ( *v19 == -2 )
          v20 = v19;
        if ( v18 == ++v19 )
        {
          if ( !v20 )
            goto LABEL_43;
          ++v15;
          *v20 = v17;
          v13 = v152;
          --i;
          v12 = v151;
          ++v150;
          if ( (__int64 *)v16 != v15 )
            goto LABEL_5;
          goto LABEL_14;
        }
      }
      continue;
    }
LABEL_43:
    if ( HIDWORD(v153) < (unsigned int)v153 )
    {
      ++HIDWORD(v153);
      *v18 = v17;
      v12 = v151;
      ++v150;
      v13 = v152;
    }
    else
    {
LABEL_3:
      sub_16CCBA0((__int64)&v150, v17);
      v13 = v152;
      v12 = v151;
    }
  }
LABEL_14:
  v134 = *(const char **)(a2 + 40);
  LODWORD(m128i_i64) = sub_14AF470(a2, 0, 0, 0);
  if ( (_BYTE)m128i_i64
    || (v159 = v134, !(unsigned __int8)sub_190CC30(a1 + 112, (__int64 *)&v159, &v172))
    || v172.m128i_i64[0] == *(_QWORD *)(a1 + 120) + 16LL * *(unsigned int *)(a1 + 136)
    || !(unsigned __int8)sub_1B29870(*(_QWORD *)(a1 + 144), *(_QWORD *)(v172.m128i_i64[0] + 8), a2) )
  {
    v22 = (__int64)v134;
    while ( sub_157F0B0(v22) )
    {
      v22 = sub_157F0B0(v22);
      if ( v134 == (const char *)v22 )
        goto LABEL_31;
      v23 = (unsigned __int64)v152;
      v24 = v151;
      if ( v152 == v151 )
      {
        v34 = &v152[HIDWORD(v153)];
        if ( v152 == v34 )
        {
          v25 = v152;
        }
        else
        {
          do
          {
            if ( v22 == *v24 )
              break;
            ++v24;
          }
          while ( v34 != v24 );
          v25 = &v152[HIDWORD(v153)];
        }
        goto LABEL_39;
      }
      v25 = &v152[(unsigned int)v153];
      v24 = sub_16CC9F0((__int64)&v150, v22);
      if ( v22 == *v24 )
      {
        v23 = (unsigned __int64)v152;
        if ( v152 == v151 )
          v34 = &v152[HIDWORD(v153)];
        else
          v34 = &v152[(unsigned int)v153];
LABEL_39:
        while ( v34 != v24 && (unsigned __int64)*v24 >= 0xFFFFFFFFFFFFFFFELL )
          ++v24;
        goto LABEL_22;
      }
      v23 = (unsigned __int64)v152;
      if ( v152 == v151 )
      {
        v24 = &v152[HIDWORD(v153)];
        v34 = v24;
        goto LABEL_39;
      }
      v24 = &v152[(unsigned int)v153];
LABEL_22:
      if ( v24 != v25 )
      {
        LODWORD(m128i_i64) = 0;
        goto LABEL_24;
      }
      v27 = sub_157EBA0(v22);
      v28 = sub_15F4D60(v27);
      if ( v28 != 1 )
        goto LABEL_31;
      if ( !(_BYTE)m128i_i64 )
      {
        v29 = *(_DWORD *)(a1 + 136);
        if ( v29 )
        {
          v30 = v29 - 1;
          v31 = *(_QWORD *)(a1 + 120);
          v32 = (v29 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
          v33 = *(_QWORD *)(v31 + 16LL * v32);
          if ( v33 == v22 )
          {
LABEL_31:
            v23 = (unsigned __int64)v152;
            LODWORD(m128i_i64) = 0;
            goto LABEL_24;
          }
          while ( v33 != -8 )
          {
            v32 = v30 & (v28 + v32);
            v33 = *(_QWORD *)(v31 + 16LL * v32);
            if ( v22 == v33 )
              goto LABEL_31;
            ++v28;
          }
        }
      }
    }
    v143 = 0;
    v144 = 0;
    v145 = 0;
    v35 = *(__int64 **)a3;
    v146 = 0;
    v36 = *(unsigned int *)(a3 + 8);
    v147 = 0;
    v148 = 0;
    v37 = 3 * v36;
    v149 = 0;
    v136 = 0;
    v137 = 0;
    v138 = 0;
    v139 = 0;
    if ( v35 != &v35[v37] )
    {
      v38 = 0;
      v39 = 0;
      v40 = &v35[v37];
      while ( 1 )
      {
        if ( v39 )
        {
          v41 = (v39 - 1) & (((unsigned int)*v35 >> 9) ^ ((unsigned int)*v35 >> 4));
          v42 = v38 + 16LL * v41;
          v43 = *(_QWORD *)v42;
          if ( *(_QWORD *)v42 == *v35 )
            goto LABEL_53;
          v80 = 0;
          v81 = 1;
          while ( v43 != -8 )
          {
            if ( !v80 && v43 == -16 )
              v80 = v42;
            v41 = (v39 - 1) & (v81 + v41);
            v42 = v38 + 16LL * v41;
            v43 = *(_QWORD *)v42;
            if ( *v35 == *(_QWORD *)v42 )
              goto LABEL_53;
            ++v81;
          }
          if ( v80 )
            v42 = v80;
          ++v136;
          v44 = v138 + 1;
          if ( 4 * ((int)v138 + 1) < 3 * v39 )
          {
            if ( v39 - (v44 + HIDWORD(v138)) > v39 >> 3 )
              goto LABEL_59;
            goto LABEL_58;
          }
        }
        else
        {
          ++v136;
        }
        v39 *= 2;
LABEL_58:
        sub_1911330((__int64)&v136, v39);
        sub_190EEA0((__int64)&v136, v35, &v172);
        v42 = v172.m128i_i64[0];
        v44 = v138 + 1;
LABEL_59:
        LODWORD(v138) = v44;
        if ( *(_QWORD *)v42 != -8 )
          --HIDWORD(v138);
        v45 = *v35;
        *(_BYTE *)(v42 + 8) = 0;
        *(_QWORD *)v42 = v45;
LABEL_53:
        v35 += 3;
        *(_BYTE *)(v42 + 8) = 1;
        if ( v35 == v40 )
          break;
        v38 = v137;
        v39 = v139;
      }
    }
    v46 = *(__int64 **)a4;
    v47 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
    if ( v47 == *(_QWORD *)a4 )
    {
LABEL_73:
      v140 = (__int64 *)v142;
      v141 = 0x400000000LL;
      v54 = *(_QWORD *)(v22 + 8);
      if ( v54 )
      {
        while ( 1 )
        {
          v55 = sub_1648700(v54);
          if ( (unsigned __int8)(*((_BYTE *)v55 + 16) - 25) <= 9u )
            break;
          v54 = *(_QWORD *)(v54 + 8);
          if ( !v54 )
            goto LABEL_118;
        }
        v56 = 0x40018000000001LL;
LABEL_77:
        v57 = v55[5];
        v172.m128i_i64[0] = v57;
        v58 = (unsigned int)*(unsigned __int8 *)(sub_157EBA0(v57) + 16) - 34;
        if ( (unsigned int)v58 <= 0x36 && _bittest64(&v56, v58) )
          goto LABEL_125;
        if ( (unsigned __int8)sub_19114F0(v57, (__int64)&v136, 0) )
          goto LABEL_89;
        v59 = sub_157EBA0(v172.m128i_i64[0]);
        if ( (unsigned int)sub_15F4D60(v59) == 1 )
        {
          *(_QWORD *)sub_190F9C0((__int64)&v143, (unsigned __int64 *)&v172) = 0;
          goto LABEL_89;
        }
        if ( *(_BYTE *)(sub_157EBA0(v172.m128i_i64[0]) + 16) == 28
          || (v60 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(v22) + 16) - 34, (unsigned int)v60 <= 0x36)
          && _bittest64(&v56, v60)
          || !(unsigned __int8)sub_190AE20(a1) && sub_15CC8F0(*(_QWORD *)(a1 + 24), v22, v172.m128i_i64[0]) )
        {
LABEL_125:
          v64 = v140;
          LODWORD(m128i_i64) = 0;
LABEL_112:
          if ( v64 != (__int64 *)v142 )
            _libc_free((unsigned __int64)v64);
          goto LABEL_114;
        }
        v63 = (unsigned int)v141;
        if ( (unsigned int)v141 >= HIDWORD(v141) )
        {
          sub_16CD150((__int64)&v140, v142, 0, 8, v61, v62);
          v63 = (unsigned int)v141;
        }
        v140[v63] = v172.m128i_i64[0];
        LODWORD(v141) = v141 + 1;
LABEL_89:
        while ( 1 )
        {
          v54 = *(_QWORD *)(v54 + 8);
          if ( !v54 )
            break;
          v55 = sub_1648700(v54);
          if ( (unsigned __int8)(*((_BYTE *)v55 + 16) - 25) <= 9u )
            goto LABEL_77;
        }
        LODWORD(m128i_i64) = 0;
        v64 = v140;
        if ( (_DWORD)v141 + (unsigned int)(((char *)v148 - (char *)v147) >> 4) != 1 )
          goto LABEL_112;
        v65 = &v140[(unsigned int)v141];
        if ( v65 != v140 )
        {
          v66 = v140;
          do
          {
            v67 = *v66++;
            v172.m128i_i64[0] = sub_190B590(a1, v67, v22);
            *(_QWORD *)sub_190F9C0((__int64)&v143, (unsigned __int64 *)&v172) = 0;
          }
          while ( v65 != v66 );
        }
      }
      else
      {
LABEL_118:
        LODWORD(m128i_i64) = 0;
        if ( (unsigned int)(((char *)v148 - (char *)v147) >> 4) != 1 )
        {
LABEL_114:
          j___libc_free_0(v137);
          if ( v147 )
            j_j___libc_free_0(v147, v149 - (_QWORD)v147);
          j___libc_free_0(v144);
          goto LABEL_49;
        }
      }
      v68 = sub_15F2050(a2);
      v130 = sub_1632FA0(v68);
      v156 = (__m128i **)v158;
      v157 = 0x800000000LL;
      v128 = v148;
      if ( v148 == v147 )
      {
        v127 = a1 + 152;
      }
      else
      {
        v69 = v147;
        m128i_i64 = &v176;
        do
        {
          v70 = *(_QWORD *)(a1 + 40);
          v71 = *v69;
          v173 = 0;
          v72 = *(_QWORD *)(a2 - 24);
          v174.m128i_i64[0] = v70;
          v172.m128i_i64[1] = v130;
          v172.m128i_i64[0] = v72;
          v174.m128i_i64[1] = (__int64)&v176;
          v175 = 0x400000000LL;
          if ( *(_BYTE *)(v72 + 16) > 0x17u )
          {
            v176 = v72;
            LODWORD(v175) = 1;
          }
          v73 = sub_143CDC0(v172.m128i_i64, v22, v71, *(_QWORD *)(a1 + 24), (__int64)&v156);
          if ( !v73 )
          {
            if ( (__int64 *)v174.m128i_i64[1] != &v176 )
              _libc_free(v174.m128i_u64[1]);
            for ( j = v157; j; ++*(_DWORD *)(a1 + 680) )
            {
              m128i_i64 = v156[j - 1]->m128i_i64;
              LODWORD(v157) = j - 1;
              sub_190ACD0(a1 + 152, (__int64)m128i_i64);
              v77 = *(unsigned int *)(a1 + 680);
              if ( (unsigned int)v77 >= *(_DWORD *)(a1 + 684) )
              {
                sub_16CD150(a1 + 672, (const void *)(a1 + 688), 0, 8, v75, v76);
                v77 = *(unsigned int *)(a1 + 680);
              }
              *(_QWORD *)(*(_QWORD *)(a1 + 672) + 8 * v77) = m128i_i64;
              j = v157;
            }
            LOBYTE(m128i_i64) = (_DWORD)v141 != 0;
LABEL_109:
            if ( v156 != (__m128i **)v158 )
              _libc_free((unsigned __int64)v156);
            v64 = v140;
            goto LABEL_112;
          }
          v69[1] = v73;
          if ( (__int64 *)v174.m128i_i64[1] != &v176 )
            _libc_free(v174.m128i_u64[1]);
          v69 += 2;
        }
        while ( v128 != v69 );
        v82 = v156;
        v83 = &v156[(unsigned int)v157];
        if ( v156 != v83 )
        {
          do
          {
            v84 = *v82;
            v172.m128i_i64[0] = 0;
            if ( &v84[3] != &v172 )
            {
              v85 = v84[3].m128i_i64[0];
              if ( v85 )
              {
                sub_161E7C0((__int64)v84[3].m128i_i64, v85);
                v86 = (unsigned __int8 *)v172.m128i_i64[0];
                v84[3].m128i_i64[0] = v172.m128i_i64[0];
                if ( v86 )
                  sub_1623210((__int64)&v172, v86, (__int64)v84[3].m128i_i64);
              }
            }
            ++v82;
            sub_1911FD0(a1 + 152, (__int64)v84);
          }
          while ( v83 != v82 );
        }
        v127 = a1 + 152;
        v125 = v148;
        if ( v148 != v147 )
        {
          v131 = v147;
          while ( 1 )
          {
            v95 = *v131;
            v126 = *v131;
            v135 = (__int64 *)v131[1];
            v159 = sub_1649960(a2);
            v96 = *(unsigned __int16 *)(a2 + 18);
            v172.m128i_i64[1] = (__int64)".pre";
            v160 = v97;
            v98 = *(_BYTE *)(a2 + 56);
            LOWORD(v173) = 773;
            v119 = v96 & 1;
            v120 = 1 << (v96 >> 1) >> 1;
            v121 = v98;
            v172.m128i_i64[0] = (__int64)&v159;
            v123 = (v96 >> 7) & 7;
            v122 = sub_157EBA0(v95);
            v99 = (__int64)sub_1648A60(64, 1u);
            if ( v99 )
              sub_15F8F80(v99, *(_QWORD *)(*v135 + 24), (__int64)v135, (__int64)&v172, v119, v120, v123, v121, v122);
            v100 = *(_QWORD *)(a2 + 48);
            v87 = (__m128i *)(v99 + 48);
            v172.m128i_i64[0] = v100;
            if ( v100 )
              break;
            if ( v87 != &v172 )
            {
              v101 = *(_QWORD *)(v99 + 48);
              if ( v101 )
                goto LABEL_175;
            }
LABEL_156:
            v159 = 0;
            v160 = 0;
            v161 = 0;
            sub_14A8180(a2, (__int64 *)&v159, 0);
            if ( v159 || v160 || v161 )
              sub_1626170(v99, (__int64 *)&v159);
            if ( *(_QWORD *)(a2 + 48) || *(__int16 *)(a2 + 18) < 0 )
            {
              v90 = sub_1625790(a2, 6);
              if ( v90 )
                sub_1625C10(v99, 6, v90);
              if ( *(_QWORD *)(a2 + 48) || *(__int16 *)(a2 + 18) < 0 )
              {
                v91 = sub_1625790(a2, 16);
                if ( v91 )
                  sub_1625C10(v99, 16, v91);
                if ( *(_QWORD *)(a2 + 48) || *(__int16 *)(a2 + 18) < 0 )
                {
                  v92 = sub_1625790(a2, 4);
                  if ( v92 )
                    sub_1625C10(v99, 4, v92);
                }
              }
            }
            LODWORD(v173) = 0;
            v172.m128i_i64[0] = v126;
            v172.m128i_i64[1] = v99 & 0xFFFFFFFFFFFFFFF9LL;
            v93 = *(unsigned int *)(a3 + 8);
            if ( (unsigned int)v93 >= *(_DWORD *)(a3 + 12) )
            {
              sub_16CD150(a3, (const void *)(a3 + 16), 0, 24, v88, v89);
              v93 = *(unsigned int *)(a3 + 8);
            }
            a5 = (__m128)_mm_loadu_si128(&v172);
            v94 = (__m128 *)(*(_QWORD *)a3 + 24 * v93);
            *v94 = a5;
            v94[1].m128_u64[0] = v173;
            ++*(_DWORD *)(a3 + 8);
            sub_14134C0(*(_QWORD *)a1, v135);
            sub_14139C0(*(_QWORD *)a1, v99);
            v131 += 2;
            if ( v125 == v131 )
              goto LABEL_189;
          }
          sub_1623A60((__int64)&v172, v100, 2);
          v87 = (__m128i *)(v99 + 48);
          if ( (__m128i *)(v99 + 48) == &v172 )
          {
            if ( v172.m128i_i64[0] )
              sub_161E7C0((__int64)&v172, v172.m128i_i64[0]);
            goto LABEL_156;
          }
          v101 = *(_QWORD *)(v99 + 48);
          if ( v101 )
          {
LABEL_175:
            v124 = v87;
            sub_161E7C0((__int64)v87, v101);
            v87 = v124;
          }
          v102 = (unsigned __int8 *)v172.m128i_i64[0];
          *(_QWORD *)(v99 + 48) = v172.m128i_i64[0];
          if ( v102 )
            sub_1623210((__int64)&v172, v102, (__int64)v87);
          goto LABEL_156;
        }
      }
LABEL_189:
      v103 = (__int64 *)sub_190AF50((__int64 ***)a2, a3, (__int64 *)a1);
      sub_164D160(a2, (__int64)v103, a5, a6, a7, a8, v104, v105, a11, a12);
      v106 = *((_BYTE *)v103 + 16);
      if ( v106 == 77 )
      {
        sub_14139C0(*(_QWORD *)a1, (__int64)v103);
        sub_164B7C0((__int64)v103, a2);
        v106 = *((_BYTE *)v103 + 16);
      }
      if ( v106 > 0x17u )
      {
        v107 = (__m128i *)(v103 + 6);
        v108 = *(_QWORD *)(a2 + 48);
        v172.m128i_i64[0] = v108;
        if ( v108 )
        {
          sub_1623A60((__int64)&v172, v108, 2);
          if ( v107 == &v172 )
            goto LABEL_195;
          goto LABEL_194;
        }
        if ( v107 != &v172 )
        {
LABEL_194:
          sub_19094F0(v103 + 6, (unsigned __int8 **)&v172);
LABEL_195:
          if ( v172.m128i_i64[0] )
            sub_161E7C0((__int64)&v172, v172.m128i_i64[0]);
        }
      }
      v109 = *(_BYTE *)(*v103 + 8);
      if ( v109 == 16 )
        v109 = *(_BYTE *)(**(_QWORD **)(*v103 + 16) + 8LL);
      if ( v109 == 15 )
        sub_14134C0(*(_QWORD *)a1, v103);
      sub_190ACD0(v127, a2);
      v112 = *(unsigned int *)(a1 + 680);
      if ( (unsigned int)v112 >= *(_DWORD *)(a1 + 684) )
      {
        sub_16CD150(a1 + 672, (const void *)(a1 + 688), 0, 8, v110, v111);
        v112 = *(unsigned int *)(a1 + 680);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 672) + 8 * v112) = a2;
      v113 = *(__int64 **)(a1 + 104);
      ++*(_DWORD *)(a1 + 680);
      v114 = sub_15E0530(*v113);
      if ( sub_1602790(v114)
        || (v117 = sub_15E0530(*v113),
            v118 = sub_16033E0(v117),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v118 + 48LL))(v118)) )
      {
        sub_15CA3B0((__int64)&v172, (__int64)"gvn", (__int64)"LoadPRE", 7, a2);
        sub_15CAB20((__int64)&v172, "load eliminated by PRE", 0x16u);
        v115 = _mm_loadu_si128(&v174);
        v116 = _mm_loadu_si128(&v177);
        LODWORD(v160) = v172.m128i_i32[2];
        v162 = v115;
        BYTE4(v160) = v172.m128i_i8[12];
        v165 = v116;
        v161 = v173;
        v163 = v175;
        v159 = (const char *)&unk_49ECF68;
        v164 = v176;
        v167 = v179;
        if ( v179 )
          v166 = v178;
        v168[0] = v169;
        v168[1] = 0x400000000LL;
        if ( v181 )
          sub_190E6E0((__int64)v168, (__int64)v180);
        v172.m128i_i64[0] = (__int64)&unk_49ECF68;
        v169[352] = v182;
        v170 = v183;
        v171 = v184;
        v159 = (const char *)&unk_49ECF98;
        sub_1897B80((__int64)v180);
        sub_143AA50(v113, (__int64)&v159);
        v159 = (const char *)&unk_49ECF68;
        sub_1897B80((__int64)v168);
      }
      LODWORD(m128i_i64) = 1;
      goto LABEL_109;
    }
    while ( 1 )
    {
      v51 = (const char *)*v46;
      v52 = v139;
      v159 = (const char *)*v46;
      if ( !v139 )
        break;
      v48 = (v139 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
      v49 = v137 + 16LL * v48;
      v50 = *(_QWORD *)v49;
      if ( v51 != *(const char **)v49 )
      {
        v78 = 0;
        v79 = 1;
        while ( v50 != -8 )
        {
          if ( v50 == -16 && !v78 )
            v78 = v49;
          v48 = (v139 - 1) & (v79 + v48);
          v49 = v137 + 16LL * v48;
          v50 = *(_QWORD *)v49;
          if ( v51 == *(const char **)v49 )
            goto LABEL_65;
          ++v79;
        }
        if ( v78 )
          v49 = v78;
        ++v136;
        v53 = v138 + 1;
        if ( 4 * ((int)v138 + 1) < 3 * v139 )
        {
          if ( v139 - HIDWORD(v138) - v53 <= v139 >> 3 )
          {
LABEL_69:
            sub_1911330((__int64)&v136, v52);
            sub_190EEA0((__int64)&v136, (__int64 *)&v159, &v172);
            v49 = v172.m128i_i64[0];
            v51 = v159;
            v53 = v138 + 1;
          }
          LODWORD(v138) = v53;
          if ( *(_QWORD *)v49 != -8 )
            --HIDWORD(v138);
          *(_QWORD *)v49 = v51;
          *(_BYTE *)(v49 + 8) = 0;
          goto LABEL_65;
        }
LABEL_68:
        v52 = 2 * v139;
        goto LABEL_69;
      }
LABEL_65:
      ++v46;
      *(_BYTE *)(v49 + 8) = 0;
      if ( (__int64 *)v47 == v46 )
        goto LABEL_73;
    }
    ++v136;
    goto LABEL_68;
  }
LABEL_49:
  v23 = (unsigned __int64)v152;
LABEL_24:
  if ( (__int64 *)v23 != v151 )
    _libc_free(v23);
  return (unsigned int)m128i_i64;
}

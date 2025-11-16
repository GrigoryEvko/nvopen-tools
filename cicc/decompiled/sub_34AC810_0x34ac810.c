// Function: sub_34AC810
// Address: 0x34ac810
//
void __fastcall sub_34AC810(__int64 a1, __int64 a2, __int64 *a3, _QWORD *a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v6; // rax
  _DWORD *v7; // rdx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int32 v10; // r14d
  __int64 v11; // r13
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned int v18; // ecx
  __int64 v19; // r9
  int v20; // eax
  __int64 v21; // r15
  unsigned int v22; // r12d
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r10
  unsigned __int64 v26; // rcx
  __int64 v27; // r11
  unsigned __int64 v28; // rax
  __int64 v29; // r8
  __int64 v30; // r9
  unsigned __int64 v31; // rdx
  const __m128i *v32; // rsi
  const __m128i *v33; // rbx
  __int64 v34; // r15
  __int64 v35; // rdx
  const __m128i *v36; // rax
  char v37; // cl
  int v38; // eax
  __m128i v39; // rax
  unsigned __int64 v40; // rbx
  unsigned __int64 v41; // rdi
  __int64 v42; // rax
  int v43; // edx
  int v44; // ecx
  int v45; // edx
  __int64 v46; // rax
  __int64 v47; // rsi
  int v48; // r15d
  __int64 v49; // rsi
  char *v50; // rax
  char *v51; // rsi
  __int64 v52; // r12
  __int64 v53; // rax
  __int64 *v54; // rax
  __int64 *v55; // rdi
  unsigned __int64 v56; // rcx
  __int64 v57; // rdx
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // rdx
  unsigned int v70; // ecx
  unsigned int v71; // r12d
  __int64 v72; // rax
  __int64 v73; // rax
  int v74; // r11d
  __int64 v75; // r10
  int v76; // r10d
  _DWORD *v77; // r11
  __int64 v78; // r12
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rdx
  unsigned __int64 v82; // rbx
  unsigned __int64 v83; // rdi
  unsigned int v84; // esi
  __int64 v85; // rax
  int v86; // edx
  int v87; // ecx
  __m128i v88; // rax
  __m128i v89; // rax
  unsigned int v90; // esi
  int v91; // ecx
  _DWORD *v92; // rax
  __int64 v93; // [rsp+18h] [rbp-668h]
  unsigned __int64 v94; // [rsp+38h] [rbp-648h]
  __int64 v95; // [rsp+40h] [rbp-640h]
  int v97; // [rsp+50h] [rbp-630h]
  __int64 v98; // [rsp+58h] [rbp-628h]
  __int64 v99; // [rsp+58h] [rbp-628h]
  int v103; // [rsp+78h] [rbp-608h]
  __int64 v104; // [rsp+78h] [rbp-608h]
  __int64 v105; // [rsp+78h] [rbp-608h]
  __int64 v106; // [rsp+80h] [rbp-600h]
  __int64 v108; // [rsp+90h] [rbp-5F0h] BYREF
  _BYTE *v109; // [rsp+98h] [rbp-5E8h] BYREF
  __int64 v110; // [rsp+A0h] [rbp-5E0h]
  _BYTE v111[64]; // [rsp+A8h] [rbp-5D8h] BYREF
  unsigned int v112; // [rsp+E8h] [rbp-598h]
  __int64 v113; // [rsp+F0h] [rbp-590h]
  unsigned __int64 v114; // [rsp+F8h] [rbp-588h]
  __int64 v115; // [rsp+100h] [rbp-580h]
  unsigned __int64 v116[2]; // [rsp+108h] [rbp-578h] BYREF
  _BYTE v117[64]; // [rsp+118h] [rbp-568h] BYREF
  int v118; // [rsp+158h] [rbp-528h]
  __int64 v119; // [rsp+160h] [rbp-520h]
  __int64 v120; // [rsp+168h] [rbp-518h]
  __int64 v121; // [rsp+170h] [rbp-510h] BYREF
  _BYTE *v122; // [rsp+178h] [rbp-508h] BYREF
  __int64 v123; // [rsp+180h] [rbp-500h]
  _BYTE v124[64]; // [rsp+188h] [rbp-4F8h] BYREF
  int v125; // [rsp+1C8h] [rbp-4B8h]
  __int64 v126; // [rsp+1D0h] [rbp-4B0h]
  unsigned __int64 v127; // [rsp+1D8h] [rbp-4A8h]
  __int64 v128; // [rsp+1E0h] [rbp-4A0h]
  unsigned __int64 v129[2]; // [rsp+1E8h] [rbp-498h] BYREF
  _BYTE v130[64]; // [rsp+1F8h] [rbp-488h] BYREF
  int v131; // [rsp+238h] [rbp-448h]
  __int64 v132; // [rsp+240h] [rbp-440h]
  __int64 v133; // [rsp+248h] [rbp-438h]
  unsigned __int64 v134[2]; // [rsp+250h] [rbp-430h] BYREF
  _BYTE v135[136]; // [rsp+260h] [rbp-420h] BYREF
  int v136; // [rsp+2E8h] [rbp-398h] BYREF
  unsigned __int64 v137; // [rsp+2F0h] [rbp-390h]
  int *v138; // [rsp+2F8h] [rbp-388h]
  int *v139; // [rsp+300h] [rbp-380h]
  __int64 v140; // [rsp+308h] [rbp-378h]
  __int64 v141; // [rsp+310h] [rbp-370h] BYREF
  char *v142; // [rsp+318h] [rbp-368h] BYREF
  int v143; // [rsp+320h] [rbp-360h]
  char v144; // [rsp+328h] [rbp-358h] BYREF
  unsigned int v145; // [rsp+368h] [rbp-318h]
  __int64 v146; // [rsp+370h] [rbp-310h]
  unsigned __int64 v147; // [rsp+378h] [rbp-308h]
  __int64 v148; // [rsp+380h] [rbp-300h]
  char *v149; // [rsp+388h] [rbp-2F8h] BYREF
  int v150; // [rsp+390h] [rbp-2F0h]
  char v151; // [rsp+398h] [rbp-2E8h] BYREF
  int v152; // [rsp+3D8h] [rbp-2A8h]
  __int64 v153; // [rsp+3E0h] [rbp-2A0h]
  __int64 v154; // [rsp+3E8h] [rbp-298h]
  __int64 v155; // [rsp+3F0h] [rbp-290h] BYREF
  _BYTE *v156; // [rsp+3F8h] [rbp-288h] BYREF
  __int64 v157; // [rsp+400h] [rbp-280h] BYREF
  _BYTE v158[64]; // [rsp+408h] [rbp-278h] BYREF
  unsigned int v159; // [rsp+448h] [rbp-238h]
  __int64 v160; // [rsp+450h] [rbp-230h]
  __int64 v161; // [rsp+458h] [rbp-228h]
  __int64 v162; // [rsp+460h] [rbp-220h]
  _BYTE *v163; // [rsp+468h] [rbp-218h] BYREF
  __int64 v164; // [rsp+470h] [rbp-210h]
  _BYTE v165[64]; // [rsp+478h] [rbp-208h] BYREF
  int v166; // [rsp+4B8h] [rbp-1C8h]
  __int64 v167; // [rsp+4C0h] [rbp-1C0h]
  __int64 v168; // [rsp+4C8h] [rbp-1B8h]
  __m128i v169; // [rsp+4D0h] [rbp-1B0h] BYREF
  __m128i v170; // [rsp+4E0h] [rbp-1A0h] BYREF
  __int64 v171; // [rsp+4F0h] [rbp-190h]
  char *v172; // [rsp+510h] [rbp-170h]
  int v173; // [rsp+518h] [rbp-168h]
  char v174; // [rsp+520h] [rbp-160h] BYREF
  unsigned int v175; // [rsp+528h] [rbp-158h]
  __int64 v176; // [rsp+530h] [rbp-150h]
  __int64 v177; // [rsp+538h] [rbp-148h]
  __int64 v178; // [rsp+540h] [rbp-140h]
  char *v179; // [rsp+548h] [rbp-138h] BYREF
  char v180; // [rsp+558h] [rbp-128h] BYREF
  int v181; // [rsp+598h] [rbp-E8h]
  __int64 v182; // [rsp+5A0h] [rbp-E0h]
  __int64 v183; // [rsp+5A8h] [rbp-D8h]
  char *v184; // [rsp+620h] [rbp-60h]
  char v185; // [rsp+630h] [rbp-50h] BYREF

  v5 = a2;
  sub_2E88D60(a2);
  v136 = 0;
  v134[0] = (unsigned __int64)v135;
  v134[1] = 0x2000000000LL;
  v138 = &v136;
  v139 = &v136;
  v6 = *(_QWORD *)(a2 + 48);
  v137 = 0;
  v140 = 0;
  v7 = (_DWORD *)(v6 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v6 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_48;
  if ( (v6 & 7) != 0 )
  {
    if ( (v6 & 7) == 3 )
    {
      if ( *v7 != 1 )
      {
        v106 = 0;
        v10 = 0;
        v11 = 0;
        goto LABEL_26;
      }
      goto LABEL_4;
    }
LABEL_48:
    v106 = 0;
    v11 = 0;
    v10 = 0;
    goto LABEL_49;
  }
  *(_QWORD *)(a2 + 48) = v7;
LABEL_4:
  v8 = sub_2E8E2F0(a2, *(_QWORD *)(a1 + 16));
  v156 = (_BYTE *)v9;
  v155 = v8;
  if ( (_BYTE)v9 || (v88.m128i_i64[0] = sub_2E8E430(a2, *(__int64 **)(a1 + 16)), v169 = v88, v88.m128i_i8[8]) )
  {
    sub_349EA70((__int64)&v169, a1, a2);
    v10 = v169.m128i_i32[0];
    v11 = v169.m128i_i64[1];
    v106 = v170.m128i_i64[0];
    sub_34A41A0((__int64)&v141, (__int64)(a3 + 1), (_QWORD *)0x4000000000000000LL, 0x4000000100000000uLL, v12, v13);
    v108 = v141;
    v109 = v111;
    v110 = 0x400000000LL;
    if ( v143 )
      sub_349DB40((__int64)&v109, (__int64)&v142, v14, v15, v16, v17);
    v18 = v145;
    v19 = v146;
    v114 = v147;
    v112 = v145;
    v115 = v148;
    v116[0] = (unsigned __int64)v117;
    v113 = v146;
    v116[1] = 0x400000000LL;
    if ( v150 )
    {
      sub_349DB40((__int64)v116, (__int64)&v149, v14, v145, v16, v146);
      v18 = v112;
      v19 = v113;
    }
    v20 = v152;
    v21 = v19;
    v22 = v18;
    v119 = v153;
    v118 = v152;
    v120 = v154;
    while ( v22 != v20 || v119 != v21 || v114 != v120 )
    {
      v23 = sub_349D6E0((__int64)a4, __ROL8__(v22 + v21, 32));
      v169.m128i_i32[2] = v10;
      v169.m128i_i32[0] = 2;
      v170.m128i_i64[0] = v11;
      v170.m128i_i64[1] = v106;
      v24 = sub_349EDA0(
              *(_QWORD *)(v23 + 64),
              *(_QWORD *)(v23 + 64) + 32LL * *(unsigned int *)(v23 + 72),
              (__int64)&v169);
      v26 = v22 + v21;
      if ( v27 != v24 )
      {
        v104 = v25;
        LODWORD(v155) = v22 + v21;
        sub_2E282C0((__int64)&v169, (__int64)v134, (unsigned int *)&v155, v26, v16);
        v45 = *(_DWORD *)(v104 + 72);
        if ( !v45 )
          goto LABEL_151;
        v46 = *(_QWORD *)(v104 + 64);
        v47 = v46 + 32LL * (unsigned int)(v45 - 1) + 32;
        while ( 1 )
        {
          if ( *(_DWORD *)v46 == 2 )
          {
            v48 = *(_DWORD *)(v46 + 8);
            if ( v48 == v10 && *(_QWORD *)(v46 + 16) == v11 && *(_QWORD *)(v46 + 24) == v106 )
              break;
          }
          v46 += 32;
          if ( v46 == v47 )
            goto LABEL_151;
        }
        v49 = v104;
        v98 = *(_QWORD *)(v46 + 16);
        v105 = *(_QWORD *)(v46 + 24);
        sub_349DE60(&v169, v49);
        v50 = v172;
        v51 = &v172[32 * v173];
        if ( v172 == v51 )
LABEL_151:
          BUG();
        while ( *(_DWORD *)v50 != 2
             || v48 != *((_DWORD *)v50 + 2)
             || v98 != *((_QWORD *)v50 + 2)
             || v105 != *((_QWORD *)v50 + 3) )
        {
          v50 += 32;
          if ( v51 == v50 )
            goto LABEL_151;
        }
        *(_DWORD *)v50 = 1;
        *((_QWORD *)v50 + 1) = 0;
        sub_34A0610(&v155, a4, (__int64)&v169);
        v52 = *(_QWORD *)(v155 + 8LL * (unsigned int)v156 - 8);
        v53 = *(unsigned int *)(a5 + 8);
        if ( v53 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
        {
          sub_C8D5F0(a5, (const void *)(a5 + 16), v53 + 1, 0x10u, v16, v19);
          v53 = *(unsigned int *)(a5 + 8);
        }
        v54 = (__int64 *)(*(_QWORD *)a5 + 16 * v53);
        *v54 = a2;
        v54[1] = v52;
        v55 = (__int64 *)v155;
        ++*(_DWORD *)(a5 + 8);
        if ( v55 != &v157 )
          _libc_free((unsigned __int64)v55);
        if ( v184 != &v185 )
          _libc_free((unsigned __int64)v184);
        if ( v172 != &v174 )
          _libc_free((unsigned __int64)v172);
        v21 = v113;
        v22 = v112;
        v26 = v112 + v113;
      }
      if ( v26 >= v114 )
      {
        v42 = (__int64)&v109[16 * (unsigned int)v110 - 16];
        v43 = *(_DWORD *)(v42 + 12) + 1;
        *(_DWORD *)(v42 + 12) = v43;
        v44 = v110;
        if ( v43 == *(_DWORD *)&v109[16 * (unsigned int)v110 - 8] )
        {
          v84 = *(_DWORD *)(v108 + 192);
          if ( v84 )
          {
            sub_F03D40((__int64 *)&v109, v84);
            v44 = v110;
          }
        }
        if ( v44 && *((_DWORD *)v109 + 3) < *((_DWORD *)v109 + 2) )
        {
          v112 = 0;
          v22 = 0;
          v21 = *(_QWORD *)sub_34A2590((__int64)&v108);
          v113 = v21;
          v114 = *(_QWORD *)sub_34A25B0((__int64)&v108);
        }
        else
        {
          v112 = -1;
          v21 = 0;
          v22 = -1;
          v113 = 0;
          v114 = 0;
        }
      }
      else
      {
        v112 = ++v22;
      }
      v20 = v118;
    }
    v5 = a2;
    if ( (_BYTE *)v116[0] != v117 )
      _libc_free(v116[0]);
    if ( v109 != v111 )
      _libc_free((unsigned __int64)v109);
    if ( v149 != &v151 )
      _libc_free((unsigned __int64)v149);
    if ( v142 != &v144 )
      _libc_free((unsigned __int64)v142);
    sub_34A9A60(a3, (__int64)v134, (__int64)a4, 0x40000000, v16, v19);
    v6 = *(_QWORD *)(a2 + 48);
    v7 = (_DWORD *)(v6 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v6 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_50;
    goto LABEL_26;
  }
  v6 = *(_QWORD *)(a2 + 48);
  v11 = 0;
  v10 = 0;
  v106 = 0;
  v7 = (_DWORD *)(v6 & 0xFFFFFFFFFFFFFFF8LL);
LABEL_49:
  if ( !v7 )
    goto LABEL_50;
LABEL_26:
  if ( (v6 & 7) == 0 )
  {
    *(_QWORD *)(v5 + 48) = v7;
    goto LABEL_28;
  }
  if ( (v6 & 7) != 3 )
  {
LABEL_42:
    if ( !v7 )
      goto LABEL_50;
    goto LABEL_43;
  }
  if ( *v7 == 1 )
  {
LABEL_28:
    v28 = sub_2E8E2F0(v5, *(_QWORD *)(a1 + 16));
    v156 = (_BYTE *)v31;
    v155 = v28;
    if ( (_BYTE)v31 || (v89.m128i_i64[0] = sub_2E8E430(v5, *(__int64 **)(a1 + 16)), v169 = v89, v89.m128i_i8[8]) )
    {
      v32 = *(const __m128i **)(v5 + 32);
      v33 = (const __m128i *)((char *)v32 + 40 * (*(_DWORD *)(v5 + 40) & 0xFFFFFF));
      if ( v32 != v33 )
      {
        while ( 1 )
        {
          v169 = _mm_loadu_si128(v32);
          v170 = _mm_loadu_si128(v32 + 1);
          v171 = v32[2].m128i_i64[0];
          if ( !(v32->m128i_i8[0] | v169.m128i_i8[3] & 0x10) )
          {
            v34 = v32->m128i_u32[2];
            if ( (v32->m128i_i8[3] & 0x40) != 0 )
              goto LABEL_86;
            if ( (_DWORD)v34 )
            {
              v35 = *(_QWORD *)(v5 + 8);
              if ( v35 != *(_QWORD *)(v5 + 24) + 48LL )
              {
                v36 = *(const __m128i **)(v35 + 32);
                v29 = (__int64)&v36->m128i_i64[5 * (*(_DWORD *)(v35 + 40) & 0xFFFFFF)];
                if ( v36 != (const __m128i *)v29 )
                  break;
              }
            }
          }
LABEL_40:
          v32 = (const __m128i *)((char *)v32 + 40);
          if ( v33 == v32 )
            goto LABEL_41;
        }
        while ( 1 )
        {
          v169 = _mm_loadu_si128(v36);
          v170 = _mm_loadu_si128(v36 + 1);
          v171 = v36[2].m128i_i64[0];
          if ( !v36->m128i_i8[0] )
          {
            v30 = v36->m128i_u32[2];
            v37 = (v36->m128i_i8[3] & 0x40) != 0;
            v169.m128i_i8[3] = (v37 << 6) | v169.m128i_i8[3] & 0xBF;
            if ( (v169.m128i_i8[3] & 0x10) == 0 && (_DWORD)v34 == (_DWORD)v30 && v37 )
              break;
          }
          v36 = (const __m128i *)((char *)v36 + 40);
          if ( (const __m128i *)v29 == v36 )
            goto LABEL_40;
        }
LABEL_86:
        v103 = 1;
        goto LABEL_87;
      }
    }
LABEL_41:
    v6 = *(_QWORD *)(v5 + 48);
    v7 = (_DWORD *)(v6 & 0xFFFFFFFFFFFFFFF8LL);
    goto LABEL_42;
  }
LABEL_43:
  v38 = v6 & 7;
  if ( v38 )
  {
    if ( v38 != 3 || *v7 != 1 )
      goto LABEL_50;
  }
  else
  {
    *(_QWORD *)(v5 + 48) = v7;
  }
  v39.m128i_i64[0] = sub_2E8E4C0(v5, *(_QWORD *)(a1 + 16));
  v169 = v39;
  if ( !v39.m128i_i8[8] )
    goto LABEL_50;
  v34 = *(unsigned int *)(*(_QWORD *)(v5 + 32) + 8LL);
  sub_349EA70((__int64)&v169, a1, v5);
  v103 = 2;
  v10 = v169.m128i_i32[0];
  v11 = v169.m128i_i64[1];
  v106 = v170.m128i_i64[0];
LABEL_87:
  v155 = 0;
  v56 = 0x4000000100000000LL;
  v57 = 0x4000000000000000LL;
  v156 = v158;
  v157 = 0x400000000LL;
  v164 = 0x400000000LL;
  v159 = -1;
  v160 = 0;
  v161 = 0;
  v162 = 0;
  v163 = v165;
  v166 = -1;
  v167 = 0;
  v168 = 0;
  if ( v103 == 1 )
  {
    v56 = (unsigned __int64)(unsigned int)(v34 + 1) << 32;
    v57 = v34 << 32;
  }
  sub_34A41A0((__int64)&v169, (__int64)(a3 + 1), (_QWORD *)v57, v56, v29, v30);
  v155 = v169.m128i_i64[0];
  sub_349DC20((__int64)&v156, (char **)&v169.m128i_i64[1], v58, v59, v60, v61);
  v159 = v175;
  v160 = v176;
  v161 = v177;
  v162 = v178;
  sub_349DC20((__int64)&v163, &v179, v62, v63, v64, v65);
  v166 = v181;
  v167 = v182;
  v168 = v183;
  if ( v179 != &v180 )
    _libc_free((unsigned __int64)v179);
  if ( (unsigned __int64 *)v169.m128i_i64[1] != &v170.m128i_u64[1] )
    _libc_free(v169.m128i_u64[1]);
  v69 = (unsigned int)v157;
  v121 = v155;
  v122 = v124;
  v123 = 0x400000000LL;
  if ( (_DWORD)v157 )
    sub_349DB40((__int64)&v122, (__int64)&v156, (unsigned int)v157, v66, v67, v68);
  v70 = v159;
  v99 = v160;
  v126 = v160;
  v125 = v159;
  v127 = v161;
  v128 = v162;
  v129[0] = (unsigned __int64)v130;
  v129[1] = 0x400000000LL;
  if ( (_DWORD)v164 )
  {
    sub_349DB40((__int64)v129, (__int64)&v163, v69, v159, v67, v68);
    v70 = v125;
    v99 = v126;
  }
  v93 = v5;
  v71 = v70;
  v97 = v166;
  v131 = v166;
  v132 = v167;
  v133 = v168;
  while ( 1 )
  {
    if ( v97 == v71 && v99 == v132 && v127 == v133 )
    {
      if ( (_BYTE *)v129[0] != v130 )
        _libc_free(v129[0]);
      if ( v122 != v124 )
        _libc_free((unsigned __int64)v122);
      sub_34A03D0((__int64)&v155);
      v82 = v137;
      while ( v82 )
      {
        sub_349E500(*(_QWORD *)(v82 + 24));
        v83 = v82;
        v82 = *(_QWORD *)(v82 + 16);
        j_j___libc_free_0(v83);
      }
      goto LABEL_52;
    }
    v94 = __ROL8__(v99 + v71, 32);
    v72 = sub_349D6E0((__int64)a4, v94);
    if ( v103 == 1 )
      break;
    v169.m128i_i32[2] = v10;
    v169.m128i_i32[0] = 2;
    v170.m128i_i64[0] = v11;
    v170.m128i_i64[1] = v106;
    v95 = *(_QWORD *)(v72 + 64);
    v73 = sub_349EDA0(v95, v95 + 32LL * *(unsigned int *)(v72 + 72), (__int64)&v169);
    if ( v75 != v73 )
    {
      v76 = v74;
      v77 = (_DWORD *)v95;
      v78 = v93;
      if ( v76 )
      {
        v79 = v95;
        v80 = 0;
        while ( *(_DWORD *)v79 != 2
             || *(_DWORD *)(v79 + 8) != v10
             || v11 != *(_QWORD *)(v79 + 16)
             || v106 != *(_QWORD *)(v79 + 24) )
        {
          v80 = (unsigned int)(v80 + 1);
          v79 += 32;
          if ( v76 == (_DWORD)v80 )
            goto LABEL_151;
        }
        goto LABEL_108;
      }
      goto LABEL_151;
    }
    if ( v99 + (unsigned __int64)v71 >= v127 )
    {
      v85 = (__int64)&v122[16 * (unsigned int)v123 - 16];
      v86 = *(_DWORD *)(v85 + 12) + 1;
      *(_DWORD *)(v85 + 12) = v86;
      v87 = v123;
      if ( v86 == *(_DWORD *)&v122[16 * (unsigned int)v123 - 8] )
      {
        v90 = *(_DWORD *)(v121 + 192);
        if ( v90 )
        {
          sub_F03D40((__int64 *)&v122, v90);
          v87 = v123;
        }
      }
      if ( v87 && *((_DWORD *)v122 + 3) < *((_DWORD *)v122 + 2) )
      {
        v71 = 0;
        v125 = 0;
        v99 = *(_QWORD *)sub_34A2590((__int64)&v121);
        v126 = v99;
        v127 = *(_QWORD *)sub_34A25B0((__int64)&v121);
        v97 = v131;
      }
      else
      {
        v125 = -1;
        v71 = -1;
        v126 = 0;
        v127 = 0;
        v97 = v131;
        v99 = 0;
      }
    }
    else
    {
      v125 = ++v71;
    }
  }
  v91 = *(_DWORD *)(v72 + 72);
  v78 = v93;
  if ( !v91 )
    goto LABEL_151;
  v77 = *(_DWORD **)(v72 + 64);
  v80 = 0;
  v92 = v77;
  while ( *v92 != 1 || (_DWORD)v34 != v92[2] )
  {
    v80 = (unsigned int)(v80 + 1);
    v92 += 8;
    if ( (_DWORD)v80 == v91 )
      goto LABEL_151;
  }
LABEL_108:
  v81 = 8 * v80;
  v169 = _mm_loadu_si128((const __m128i *)&v77[v81]);
  v170 = _mm_loadu_si128((const __m128i *)&v77[v81 + 4]);
  sub_34AC3C0(a1, v78, (__int64)a3, a5, (__int64)a4, v94, v103, (__int64)&v169, v34);
  if ( (_BYTE *)v129[0] != v130 )
    _libc_free(v129[0]);
  if ( v122 != v124 )
    _libc_free((unsigned __int64)v122);
  if ( v163 != v165 )
    _libc_free((unsigned __int64)v163);
  if ( v156 != v158 )
    _libc_free((unsigned __int64)v156);
LABEL_50:
  v40 = v137;
  while ( v40 )
  {
    sub_349E500(*(_QWORD *)(v40 + 24));
    v41 = v40;
    v40 = *(_QWORD *)(v40 + 16);
    j_j___libc_free_0(v41);
  }
LABEL_52:
  if ( (_BYTE *)v134[0] != v135 )
    _libc_free(v134[0]);
}

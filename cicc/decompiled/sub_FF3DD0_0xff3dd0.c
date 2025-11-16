// Function: sub_FF3DD0
// Address: 0xff3dd0
//
__int64 __fastcall sub_FF3DD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 *v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  __int64 v9; // r12
  int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 *v23; // r14
  __int64 v24; // rdx
  __int64 v25; // r8
  __int64 v26; // rdx
  __int64 v27; // r9
  __int64 v28; // r8
  __int64 v29; // rsi
  __int64 v30; // r8
  __int64 v31; // r9
  _BYTE *v32; // r13
  __int64 v33; // r15
  int v34; // r12d
  unsigned int v35; // edx
  __int64 v36; // r15
  unsigned int v37; // eax
  __int64 *v38; // r13
  __int64 v39; // r14
  __int64 v40; // rdi
  unsigned int v41; // edx
  __int64 v42; // rcx
  __int64 v43; // rcx
  __int64 v44; // rbx
  int v45; // esi
  unsigned __int64 v46; // rdx
  __int64 v47; // r15
  unsigned int v48; // ebx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  _BYTE *v52; // rax
  __m128i v53; // xmm0
  int v54; // r13d
  int v55; // ecx
  unsigned int i; // eax
  __int64 v57; // rdi
  int v58; // eax
  int v59; // ecx
  int v60; // edx
  unsigned __int64 v61; // rdi
  unsigned int j; // eax
  __int64 *v63; // rbx
  int v64; // eax
  __int64 v65; // rdx
  __int64 *v66; // r12
  __int64 *v67; // rbx
  __int64 result; // rax
  __int64 *v69; // rdi
  __int64 v70; // rax
  __int64 *v71; // rax
  __int64 v72; // rbx
  __int64 *v73; // r14
  unsigned int v74; // r13d
  __int64 *v75; // r15
  __int64 v76; // rax
  __int64 v77; // rcx
  __int64 v78; // rsi
  int v79; // edx
  __int64 v80; // r9
  __int64 v81; // r8
  unsigned int n; // eax
  __int64 v83; // rdi
  __int64 v84; // r10
  int v85; // eax
  unsigned int v86; // edx
  unsigned int v87; // eax
  unsigned int v88; // esi
  unsigned int v89; // edi
  unsigned __int64 v90; // rcx
  unsigned __int64 v91; // rcx
  unsigned int v92; // eax
  int v93; // esi
  int v94; // eax
  __int64 *v95; // rcx
  int v96; // edx
  int v97; // edi
  __int64 *v98; // rsi
  unsigned int m; // eax
  __int64 v100; // r8
  unsigned int v101; // eax
  __int64 *v102; // rcx
  int v103; // edx
  int v104; // edi
  unsigned int k; // eax
  __int64 v106; // r8
  unsigned int v107; // eax
  int v108; // edx
  __int64 v109; // rsi
  unsigned int jj; // eax
  int v111; // eax
  int v112; // edx
  unsigned int ii; // eax
  int v114; // eax
  int v115; // edx
  int v116; // edx
  __int64 v117; // [rsp-10h] [rbp-D50h]
  int v118; // [rsp+30h] [rbp-D10h]
  int v121; // [rsp+50h] [rbp-CF0h]
  __int64 *v122; // [rsp+50h] [rbp-CF0h]
  unsigned int v123; // [rsp+60h] [rbp-CE0h]
  _BYTE *v124; // [rsp+68h] [rbp-CD8h]
  char v125; // [rsp+68h] [rbp-CD8h]
  char v126; // [rsp+68h] [rbp-CD8h]
  _BYTE *v127; // [rsp+70h] [rbp-CD0h] BYREF
  __int64 v128; // [rsp+78h] [rbp-CC8h]
  _BYTE v129[64]; // [rsp+80h] [rbp-CC0h] BYREF
  _BYTE *v130; // [rsp+C0h] [rbp-C80h] BYREF
  __int64 v131; // [rsp+C8h] [rbp-C78h]
  _BYTE v132[64]; // [rsp+D0h] [rbp-C70h] BYREF
  _BYTE *v133; // [rsp+110h] [rbp-C30h] BYREF
  __int64 v134; // [rsp+118h] [rbp-C28h]
  _BYTE v135[192]; // [rsp+120h] [rbp-C20h] BYREF
  __int64 v136; // [rsp+1E0h] [rbp-B60h] BYREF
  __int64 v137; // [rsp+1E8h] [rbp-B58h]
  __int64 *v138; // [rsp+1F0h] [rbp-B50h] BYREF
  unsigned int v139; // [rsp+1F8h] [rbp-B48h]
  char v140[8]; // [rsp+2F0h] [rbp-A50h] BYREF
  char *v141; // [rsp+2F8h] [rbp-A48h]
  int v142; // [rsp+300h] [rbp-A40h]
  char v143; // [rsp+30Ch] [rbp-A34h]
  char v144; // [rsp+310h] [rbp-A30h] BYREF
  _BYTE *v145; // [rsp+350h] [rbp-9F0h] BYREF
  int v146; // [rsp+358h] [rbp-9E8h]
  int v147; // [rsp+35Ch] [rbp-9E4h]
  _BYTE v148[320]; // [rsp+360h] [rbp-9E0h] BYREF
  __int64 v149; // [rsp+4A0h] [rbp-8A0h] BYREF
  __int64 *v150; // [rsp+4A8h] [rbp-898h]
  int v151; // [rsp+4B0h] [rbp-890h]
  int v152; // [rsp+4B4h] [rbp-88Ch]
  int v153; // [rsp+4B8h] [rbp-888h]
  char v154; // [rsp+4BCh] [rbp-884h]
  __int64 v155; // [rsp+4C0h] [rbp-880h] BYREF
  __int64 *v156; // [rsp+500h] [rbp-840h] BYREF
  __int64 v157; // [rsp+508h] [rbp-838h]
  __int64 v158; // [rsp+510h] [rbp-830h] BYREF
  int v159; // [rsp+518h] [rbp-828h]
  __int64 v160; // [rsp+520h] [rbp-820h]
  int v161; // [rsp+528h] [rbp-818h]
  __int64 v162; // [rsp+530h] [rbp-810h]
  __m128i *v163; // [rsp+650h] [rbp-6F0h] BYREF
  _QWORD *v164; // [rsp+658h] [rbp-6E8h]
  char v165; // [rsp+66Ch] [rbp-6D4h]
  _BYTE v166[64]; // [rsp+670h] [rbp-6D0h] BYREF
  _BYTE *v167; // [rsp+6B0h] [rbp-690h] BYREF
  __int64 v168; // [rsp+6B8h] [rbp-688h]
  _BYTE v169[320]; // [rsp+6C0h] [rbp-680h] BYREF
  __m128i v170; // [rsp+800h] [rbp-540h] BYREF
  __int64 v171; // [rsp+810h] [rbp-530h]
  char v172; // [rsp+81Ch] [rbp-524h]
  char v173[64]; // [rsp+820h] [rbp-520h] BYREF
  _QWORD v174[2]; // [rsp+860h] [rbp-4E0h] BYREF
  _BYTE v175[320]; // [rsp+870h] [rbp-4D0h] BYREF
  _QWORD v176[3]; // [rsp+9B0h] [rbp-390h] BYREF
  char v177; // [rsp+9CCh] [rbp-374h]
  _BYTE v178[64]; // [rsp+9D0h] [rbp-370h] BYREF
  _BYTE *v179; // [rsp+A10h] [rbp-330h] BYREF
  __int64 v180; // [rsp+A18h] [rbp-328h]
  _BYTE v181[320]; // [rsp+A20h] [rbp-320h] BYREF
  __m128i *v182; // [rsp+B60h] [rbp-1E0h] BYREF
  __int64 v183; // [rsp+B68h] [rbp-1D8h]
  _BYTE v184[16]; // [rsp+B70h] [rbp-1D0h] BYREF
  char v185[64]; // [rsp+B80h] [rbp-1C0h] BYREF
  _QWORD v186[2]; // [rsp+BC0h] [rbp-180h] BYREF
  _BYTE v187[368]; // [rsp+BD0h] [rbp-170h] BYREF

  v4 = a1;
  v127 = v129;
  v128 = 0x800000000LL;
  v134 = 0x800000000LL;
  v5 = (__int64 *)&v138;
  v133 = v135;
  v136 = 0;
  v137 = 1;
  do
  {
    *v5 = -4096;
    v5 += 8;
    *((_DWORD *)v5 - 14) = 0x7FFFFFFF;
  }
  while ( v5 != (__int64 *)v140 );
  v6 = *(_QWORD *)(a2 + 80);
  v130 = v132;
  memset(v5, 0, 0x1B0u);
  v147 = 8;
  v141 = &v144;
  v145 = v148;
  if ( v6 )
    v6 -= 24;
  v151 = 8;
  v150 = &v155;
  v153 = 0;
  v154 = 1;
  v156 = &v158;
  v157 = 0x800000000LL;
  v152 = 1;
  v155 = v6;
  v149 = 1;
  v7 = *(_QWORD *)(v6 + 48);
  v131 = 0x800000000LL;
  v8 = v7 & 0xFFFFFFFFFFFFFFF8LL;
  v143 = 1;
  v142 = 8;
  if ( v8 == v6 + 48 )
    goto LABEL_166;
  if ( !v8 )
LABEL_57:
    BUG();
  v9 = v8 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 > 0xA )
  {
LABEL_166:
    v10 = 0;
    v11 = 0;
    v9 = 0;
  }
  else
  {
    v10 = sub_B46E30(v9);
    v11 = v9;
  }
  v162 = v6;
  v158 = v11;
  v159 = v10;
  v160 = v9;
  v161 = 0;
  LODWORD(v157) = 1;
  sub_E36520((__int64)&v149);
  sub_C8CD80((__int64)v176, (__int64)v178, (__int64)v140, v12, v13, v14);
  v179 = v181;
  v180 = 0x800000000LL;
  if ( v146 )
    sub_FF16F0((__int64)&v179, (__int64 *)&v145, v15, v16, v17, v18);
  sub_C8CF70((__int64)&v182, v185, 8, (__int64)v178, (__int64)v176);
  v186[0] = v187;
  v186[1] = 0x800000000LL;
  if ( (_DWORD)v180 )
    sub_FF1510((__int64)v186, (__int64)&v179, v19, v20, v21, v22);
  v23 = (__int64 *)&v163;
  sub_C8CD80((__int64)&v163, (__int64)v166, (__int64)&v149, (__int64)v166, v21, v22);
  v167 = v169;
  v168 = 0x800000000LL;
  if ( (_DWORD)v157 )
    sub_FF16F0((__int64)&v167, (__int64 *)&v156, v24, (__int64)v166, v25, (unsigned int)v157);
  sub_C8CF70((__int64)&v170, v173, 8, (__int64)v166, (__int64)&v163);
  v28 = (unsigned int)v168;
  v174[0] = v175;
  v174[1] = 0x800000000LL;
  if ( (_DWORD)v168 )
    sub_FF1510((__int64)v174, (__int64)&v167, v26, (__int64)v175, (unsigned int)v168, v27);
  v29 = (__int64)&v182;
  sub_E36A10((__int64)&v170, (__int64)&v182, (__int64)&v130, (__int64)v175, v28, v27);
  if ( (_BYTE *)v174[0] != v175 )
    _libc_free(v174[0], &v182);
  if ( !v172 )
    _libc_free(v170.m128i_i64[1], &v182);
  if ( v167 != v169 )
    _libc_free(v167, &v182);
  if ( !v165 )
    _libc_free(v164, &v182);
  if ( (_BYTE *)v186[0] != v187 )
    _libc_free(v186[0], &v182);
  if ( !v184[12] )
    _libc_free(v183, &v182);
  if ( v179 != v181 )
    _libc_free(v179, &v182);
  if ( !v177 )
    _libc_free(v176[1], &v182);
  if ( v156 != &v158 )
    _libc_free(v156, &v182);
  if ( !v154 )
    _libc_free(v150, &v182);
  if ( v145 != v148 )
    _libc_free(v145, &v182);
  if ( !v143 )
    _libc_free(v141, &v182);
  v124 = v130;
  if ( v130 != &v130[8 * (unsigned int)v131] )
  {
    v32 = &v130[8 * (unsigned int)v131];
    do
    {
      while ( 1 )
      {
        v33 = *((_QWORD *)v32 - 1);
        v29 = v33;
        v176[0] = sub_FEF800(a1, v33);
        if ( BYTE4(v176[0]) )
          break;
        v32 -= 8;
        if ( v124 == v32 )
          goto LABEL_46;
      }
      v34 = v176[0];
      v32 -= 8;
      sub_FEF2D0((__int64)&v182, v33, *(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80));
      sub_FF2EB0(a1, &v182, a3, a4, v34, (__int64)&v127, (__int64)&v133);
      v29 = v117;
    }
    while ( v124 != v32 );
LABEL_46:
    v4 = a1;
    v23 = (__int64 *)&v163;
  }
  v35 = v134;
  v36 = v4;
LABEL_48:
  if ( v35 )
    goto LABEL_73;
  v37 = v128;
  if ( (_DWORD)v128 )
  {
    v38 = v23;
    v39 = v36;
    while ( 1 )
    {
      v43 = v37--;
      v44 = *(_QWORD *)&v127[8 * v43 - 8];
      LODWORD(v128) = v37;
      if ( (*(_BYTE *)(v39 + 96) & 1) != 0 )
        break;
      v45 = *(_DWORD *)(v39 + 112);
      v40 = *(_QWORD *)(v39 + 104);
      if ( v45 )
      {
        v29 = (unsigned int)(v45 - 1);
LABEL_52:
        v41 = v29 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
        v42 = *(_QWORD *)(v40 + 16LL * v41);
        if ( v44 != v42 )
        {
          v30 = 1;
          while ( v42 != -4096 )
          {
            v31 = (unsigned int)(v30 + 1);
            v41 = v29 & (v30 + v41);
            v42 = *(_QWORD *)(v40 + 16LL * v41);
            if ( v44 == v42 )
              goto LABEL_53;
            v30 = (unsigned int)v31;
          }
          goto LABEL_60;
        }
LABEL_53:
        if ( !v37 )
          goto LABEL_72;
      }
      else
      {
LABEL_60:
        v29 = v44;
        sub_FEF2D0((__int64)&v170, v44, *(_QWORD *)(v39 + 72), *(_QWORD *)(v39 + 80));
        v46 = *(_QWORD *)(v44 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v46 != v44 + 48 )
        {
          if ( !v46 )
            goto LABEL_57;
          v47 = v46 - 24;
          if ( (unsigned int)*(unsigned __int8 *)(v46 - 24) - 30 <= 0xA )
          {
            v121 = sub_B46E30(v46 - 24);
            v182 = (__m128i *)v184;
            v183 = 0x400000000LL;
            if ( v121 )
            {
              v123 = 0;
              v48 = 0;
              v125 = 0;
              while ( 1 )
              {
                v49 = sub_B46EC0(v47, v48);
                sub_FEF2D0((__int64)v176, v49, *(_QWORD *)(v39 + 72), *(_QWORD *)(v39 + 80));
                v29 = (__int64)v38;
                v164 = v176;
                v163 = &v170;
                v50 = sub_FEF7A0(v39, v38);
                v149 = v50;
                if ( !BYTE4(v50) )
                  break;
                if ( v125 != 1 || v123 < (unsigned int)v50 )
                {
                  v125 = BYTE4(v50);
                  v123 = v50;
                }
                if ( v121 == ++v48 )
                {
                  if ( !v125 )
                    break;
                  v29 = (__int64)&v170;
                  sub_FF2EB0(v39, (__m128i **)&v170, a3, a4, v123, (__int64)&v127, (__int64)&v133);
                  v37 = v128;
                  goto LABEL_71;
                }
              }
            }
          }
        }
        v37 = v128;
LABEL_71:
        if ( !v37 )
        {
LABEL_72:
          v35 = v134;
          v36 = v39;
          v23 = v38;
          if ( !(_DWORD)v134 )
            goto LABEL_100;
LABEL_73:
          v51 = v35--;
          v52 = &v133[24 * v51];
          v53 = _mm_loadu_si128((const __m128i *)(v52 - 24));
          LODWORD(v134) = v35;
          v170 = v53;
          v171 = *((_QWORD *)v52 - 1);
          v54 = v171;
          if ( (*(_BYTE *)(v36 + 176) & 1) != 0 )
          {
            v29 = v36 + 184;
            v55 = 3;
LABEL_75:
            v31 = 1;
            for ( i = v55
                    & (((0xBF58476D1CE4E5B9LL
                       * ((unsigned int)(37 * v171)
                        | ((unsigned __int64)(((unsigned __int32)v53.m128i_i32[2] >> 9)
                                            ^ ((unsigned __int32)v53.m128i_i32[2] >> 4)) << 32))) >> 31)
                     ^ (756364221 * v171)); ; i = v55 & v58 )
            {
              v57 = v29 + 24LL * i;
              v30 = *(_QWORD *)v57;
              if ( v53.m128i_i64[1] == *(_QWORD *)v57 && (_DWORD)v171 == *(_DWORD *)(v57 + 8) )
                break;
              if ( v30 == -4096 && *(_DWORD *)(v57 + 8) == 0x7FFFFFFF )
                goto LABEL_86;
              v58 = v31 + i;
              v31 = (unsigned int)(v31 + 1);
            }
            goto LABEL_48;
          }
          v59 = *(_DWORD *)(v36 + 192);
          v29 = *(_QWORD *)(v36 + 184);
          if ( v59 )
          {
            v55 = v59 - 1;
            goto LABEL_75;
          }
LABEL_86:
          if ( (v137 & 1) != 0 )
          {
            v29 = (__int64)&v138;
            v60 = 3;
          }
          else
          {
            v65 = v139;
            v29 = (__int64)v138;
            if ( !v139 )
            {
              v87 = v137;
              ++v136;
              v63 = 0;
              v88 = ((unsigned int)v137 >> 1) + 1;
LABEL_146:
              v89 = 3 * v65;
              goto LABEL_147;
            }
            v60 = v139 - 1;
          }
          v30 = 1;
          v61 = 0;
          for ( j = v60
                  & (((0xBF58476D1CE4E5B9LL
                     * ((unsigned int)(37 * v171)
                      | ((unsigned __int64)(((unsigned __int32)v53.m128i_i32[2] >> 9)
                                          ^ ((unsigned __int32)v53.m128i_i32[2] >> 4)) << 32))) >> 31)
                   ^ (756364221 * v171)); ; j = v60 & v64 )
          {
            v63 = (__int64 *)(v29 + ((unsigned __int64)j << 6));
            v31 = *v63;
            if ( v53.m128i_i64[1] == *v63 && (_DWORD)v171 == *((_DWORD *)v63 + 2) )
              goto LABEL_123;
            if ( v31 == -4096 )
            {
              if ( *((_DWORD *)v63 + 2) == 0x7FFFFFFF )
              {
                v87 = v137;
                if ( v61 )
                  v63 = (__int64 *)v61;
                ++v136;
                v88 = ((unsigned int)v137 >> 1) + 1;
                if ( (v137 & 1) == 0 )
                {
                  v65 = v139;
                  goto LABEL_146;
                }
                v89 = 12;
                v65 = 4;
LABEL_147:
                v90 = 4 * v88;
                if ( v89 <= (unsigned int)v90 )
                {
                  sub_FF33E0((__int64)&v136, (unsigned int)(2 * v65), v65, v90, v30, v31);
                  if ( (v137 & 1) != 0 )
                  {
                    v102 = (__int64 *)&v138;
                    v103 = 3;
                  }
                  else
                  {
                    v102 = v138;
                    if ( !v139 )
                    {
LABEL_248:
                      LODWORD(v137) = (2 * ((unsigned int)v137 >> 1) + 2) | v137 & 1;
                      BUG();
                    }
                    v103 = v139 - 1;
                  }
                  v104 = 1;
                  v98 = 0;
                  for ( k = v103
                          & (((0xBF58476D1CE4E5B9LL
                             * ((unsigned int)(37 * v54)
                              | ((unsigned __int64)(((unsigned __int32)v53.m128i_i32[2] >> 9)
                                                  ^ ((unsigned __int32)v53.m128i_i32[2] >> 4)) << 32))) >> 31)
                           ^ (756364221 * v54)); ; k = v103 & v107 )
                  {
                    v63 = &v102[8 * (unsigned __int64)k];
                    v106 = *v63;
                    if ( v53.m128i_i64[1] == *v63 && v54 == *((_DWORD *)v63 + 2) )
                      break;
                    if ( v106 == -4096 )
                    {
                      if ( *((_DWORD *)v63 + 2) == 0x7FFFFFFF )
                      {
LABEL_242:
                        if ( v98 )
                          v63 = v98;
                        goto LABEL_204;
                      }
                    }
                    else if ( v106 == -8192 && *((_DWORD *)v63 + 2) == 0x80000000 && !v98 )
                    {
                      v98 = &v102[8 * (unsigned __int64)k];
                    }
                    v107 = v104 + k;
                    ++v104;
                  }
                  goto LABEL_204;
                }
                v91 = (_DWORD)v65 - HIDWORD(v137) - v88;
                if ( (unsigned int)v91 > (unsigned int)v65 >> 3 )
                  goto LABEL_149;
                sub_FF33E0((__int64)&v136, (unsigned int)v65, v65, v91, v30, v31);
                if ( (v137 & 1) != 0 )
                {
                  v95 = (__int64 *)&v138;
                  v96 = 3;
                  goto LABEL_177;
                }
                v95 = v138;
                if ( !v139 )
                  goto LABEL_248;
                v96 = v139 - 1;
LABEL_177:
                v97 = 1;
                v98 = 0;
                for ( m = v96
                        & (((0xBF58476D1CE4E5B9LL
                           * ((unsigned int)(37 * v54)
                            | ((unsigned __int64)(((unsigned __int32)v53.m128i_i32[2] >> 9)
                                                ^ ((unsigned __int32)v53.m128i_i32[2] >> 4)) << 32))) >> 31)
                         ^ (756364221 * v54)); ; m = v96 & v101 )
                {
                  v63 = &v95[8 * (unsigned __int64)m];
                  v100 = *v63;
                  if ( v53.m128i_i64[1] == *v63 && v54 == *((_DWORD *)v63 + 2) )
                    break;
                  if ( v100 == -4096 )
                  {
                    if ( *((_DWORD *)v63 + 2) == 0x7FFFFFFF )
                      goto LABEL_242;
                  }
                  else if ( v100 == -8192 && *((_DWORD *)v63 + 2) == 0x80000000 && !v98 )
                  {
                    v98 = &v95[8 * (unsigned __int64)m];
                  }
                  v101 = v97 + m;
                  ++v97;
                }
LABEL_204:
                v87 = v137;
LABEL_149:
                LODWORD(v137) = (2 * (v87 >> 1) + 2) | v87 & 1;
                if ( *v63 != -4096 || *((_DWORD *)v63 + 2) != 0x7FFFFFFF )
                  --HIDWORD(v137);
                *((_DWORD *)v63 + 2) = v54;
                v29 = (__int64)&v170;
                *v63 = v53.m128i_i64[1];
                v63[2] = (__int64)(v63 + 4);
                v63[3] = 0x400000000LL;
                sub_FEF550(v36, (__int64)&v170, (__int64)(v63 + 2));
LABEL_123:
                v30 = v63[2];
                v70 = *((unsigned int *)v63 + 6);
                v126 = 0;
                v182 = (__m128i *)v184;
                v122 = (__int64 *)(v30 + 8 * v70);
                v183 = 0x400000000LL;
                if ( (__int64 *)v30 == v122 )
                  goto LABEL_131;
                v71 = v23;
                v118 = v54;
                v72 = v36;
                v73 = (__int64 *)v30;
                v74 = 0;
                v75 = v71;
                do
                {
                  sub_FEF2D0((__int64)v176, *v73, *(_QWORD *)(v72 + 72), *(_QWORD *)(v72 + 80));
                  v29 = (__int64)v75;
                  v164 = v176;
                  v163 = &v170;
                  v76 = sub_FEF7A0(v72, v75);
                  v149 = v76;
                  if ( !BYTE4(v76) )
                  {
                    v23 = v75;
                    v36 = v72;
                    goto LABEL_131;
                  }
                  if ( (unsigned int)v76 > v74 || v126 != 1 )
                  {
                    v126 = BYTE4(v76);
                    v74 = v76;
                  }
                  ++v73;
                }
                while ( v122 != v73 );
                v23 = v75;
                v36 = v72;
                if ( v126 )
                {
                  v77 = *(_BYTE *)(v72 + 176) & 1;
                  if ( (*(_BYTE *)(v72 + 176) & 1) != 0 )
                  {
                    v78 = v72 + 184;
                    v79 = 3;
                  }
                  else
                  {
                    v86 = *(_DWORD *)(v72 + 192);
                    v78 = *(_QWORD *)(v72 + 184);
                    if ( !v86 )
                    {
                      v92 = *(_DWORD *)(v72 + 176);
                      ++*(_QWORD *)(v72 + 168);
                      v83 = 0;
                      v93 = (v92 >> 1) + 1;
                      goto LABEL_156;
                    }
                    v79 = v86 - 1;
                  }
                  v80 = 1;
                  v81 = 0;
                  for ( n = v79
                          & (((0xBF58476D1CE4E5B9LL
                             * ((unsigned int)(37 * v118)
                              | ((unsigned __int64)(((unsigned __int32)v53.m128i_i32[2] >> 9)
                                                  ^ ((unsigned __int32)v53.m128i_i32[2] >> 4)) << 32))) >> 31)
                           ^ (756364221 * v118)); ; n = v79 & v85 )
                  {
                    v83 = v78 + 24LL * n;
                    v84 = *(_QWORD *)v83;
                    if ( v53.m128i_i64[1] == *(_QWORD *)v83 && v118 == *(_DWORD *)(v83 + 8) )
                      goto LABEL_154;
                    if ( v84 == -4096 )
                    {
                      if ( *(_DWORD *)(v83 + 8) == 0x7FFFFFFF )
                      {
                        v92 = *(_DWORD *)(v72 + 176);
                        if ( v81 )
                          v83 = v81;
                        ++*(_QWORD *)(v72 + 168);
                        v93 = (v92 >> 1) + 1;
                        if ( (_BYTE)v77 )
                        {
                          v80 = 12;
                          v86 = 4;
LABEL_157:
                          v81 = v72 + 168;
                          if ( (unsigned int)v80 > 4 * v93 )
                          {
                            v77 = v86 - *(_DWORD *)(v72 + 180) - v93;
                            if ( (unsigned int)v77 > v86 >> 3 )
                              goto LABEL_159;
                            sub_FF3840((const __m128i *)(v72 + 168), v86);
                            if ( (*(_BYTE *)(v72 + 176) & 1) != 0 )
                            {
                              v77 = v72 + 184;
                              v112 = 3;
LABEL_219:
                              v81 = 1;
                              v109 = 0;
                              for ( ii = v112
                                       & (((0xBF58476D1CE4E5B9LL
                                          * ((unsigned int)(37 * v118)
                                           | ((unsigned __int64)(((unsigned __int32)v53.m128i_i32[2] >> 9)
                                                               ^ ((unsigned __int32)v53.m128i_i32[2] >> 4)) << 32))) >> 31)
                                        ^ (756364221 * v118)); ; ii = v112 & v114 )
                              {
                                v83 = v77 + 24LL * ii;
                                v80 = *(_QWORD *)v83;
                                if ( v53.m128i_i64[1] == *(_QWORD *)v83 && v118 == *(_DWORD *)(v83 + 8) )
                                  break;
                                if ( v80 == -4096 )
                                {
                                  if ( *(_DWORD *)(v83 + 8) == 0x7FFFFFFF )
                                    goto LABEL_234;
                                }
                                else if ( v80 == -8192 && *(_DWORD *)(v83 + 8) == 0x80000000 && !v109 )
                                {
                                  v109 = v77 + 24LL * ii;
                                }
                                v114 = v81 + ii;
                                v81 = (unsigned int)(v81 + 1);
                              }
LABEL_230:
                              v92 = *(_DWORD *)(v72 + 176);
LABEL_159:
                              *(_DWORD *)(v72 + 176) = (2 * (v92 >> 1) + 2) | v92 & 1;
                              if ( *(_QWORD *)v83 != -4096 || *(_DWORD *)(v83 + 8) != 0x7FFFFFFF )
                                --*(_DWORD *)(v72 + 180);
                              *(_DWORD *)(v83 + 8) = v118;
                              *(_QWORD *)v83 = v53.m128i_i64[1];
                              v94 = 1;
                              if ( v74 )
                                v94 = v74;
                              *(_DWORD *)(v83 + 16) = v94;
LABEL_154:
                              v29 = (__int64)&v170;
                              sub_FEF430(v72, (__int64)&v170, (__int64)&v127, v77, v81, v80);
                              break;
                            }
                            v115 = *(_DWORD *)(v72 + 192);
                            v77 = *(_QWORD *)(v72 + 184);
                            if ( v115 )
                            {
                              v112 = v115 - 1;
                              goto LABEL_219;
                            }
LABEL_247:
                            *(_DWORD *)(v72 + 176) = (2 * (*(_DWORD *)(v72 + 176) >> 1) + 2)
                                                   | *(_DWORD *)(v72 + 176) & 1;
                            BUG();
                          }
                          sub_FF3840((const __m128i *)(v72 + 168), 2 * v86);
                          if ( (*(_BYTE *)(v72 + 176) & 1) != 0 )
                          {
                            v77 = v72 + 184;
                            v108 = 3;
                          }
                          else
                          {
                            v116 = *(_DWORD *)(v72 + 192);
                            v77 = *(_QWORD *)(v72 + 184);
                            if ( !v116 )
                              goto LABEL_247;
                            v108 = v116 - 1;
                          }
                          v81 = 1;
                          v109 = 0;
                          for ( jj = v108
                                   & (((0xBF58476D1CE4E5B9LL
                                      * ((unsigned int)(37 * v118)
                                       | ((unsigned __int64)(((unsigned __int32)v53.m128i_i32[2] >> 9)
                                                           ^ ((unsigned __int32)v53.m128i_i32[2] >> 4)) << 32))) >> 31)
                                    ^ (756364221 * v118)); ; jj = v108 & v111 )
                          {
                            v83 = v77 + 24LL * jj;
                            v80 = *(_QWORD *)v83;
                            if ( v53.m128i_i64[1] == *(_QWORD *)v83 && v118 == *(_DWORD *)(v83 + 8) )
                              break;
                            if ( v80 == -4096 )
                            {
                              if ( *(_DWORD *)(v83 + 8) == 0x7FFFFFFF )
                              {
LABEL_234:
                                if ( v109 )
                                  v83 = v109;
                                goto LABEL_230;
                              }
                            }
                            else if ( v80 == -8192 && *(_DWORD *)(v83 + 8) == 0x80000000 && !v109 )
                            {
                              v109 = v77 + 24LL * jj;
                            }
                            v111 = v81 + jj;
                            v81 = (unsigned int)(v81 + 1);
                          }
                          goto LABEL_230;
                        }
                        v86 = *(_DWORD *)(v72 + 192);
LABEL_156:
                        v80 = 3 * v86;
                        goto LABEL_157;
                      }
                    }
                    else if ( v84 == -8192 && *(_DWORD *)(v83 + 8) == 0x80000000 && !v81 )
                    {
                      v81 = v78 + 24LL * n;
                    }
                    v85 = v80 + n;
                    v80 = (unsigned int)(v80 + 1);
                  }
                }
LABEL_131:
                v35 = v134;
                goto LABEL_48;
              }
            }
            else if ( v31 == -8192 && *((_DWORD *)v63 + 2) == 0x80000000 && !v61 )
            {
              v61 = v29 + ((unsigned __int64)j << 6);
            }
            v64 = v30 + j;
            v30 = (unsigned int)(v30 + 1);
          }
        }
      }
    }
    v40 = v39 + 104;
    v29 = 3;
    goto LABEL_52;
  }
LABEL_100:
  if ( v130 != v132 )
    _libc_free(v130, v29);
  if ( (v137 & 1) != 0 )
  {
    v67 = (__int64 *)v140;
    v66 = (__int64 *)&v138;
    goto LABEL_110;
  }
  v66 = v138;
  v29 = (unsigned __int64)v139 << 6;
  if ( !v139 || (v67 = (__int64 *)((char *)v138 + v29), v138 == (__int64 *)((char *)v138 + v29)) )
  {
    result = sub_C7D6A0((__int64)v138, v29, 8);
    goto LABEL_115;
  }
LABEL_110:
  while ( 2 )
  {
    while ( 2 )
    {
      result = *v66;
      if ( *v66 != -4096 )
      {
        if ( result == -8192 && *((_DWORD *)v66 + 2) == 0x80000000 )
          goto LABEL_109;
        goto LABEL_107;
      }
      if ( *((_DWORD *)v66 + 2) != 0x7FFFFFFF )
      {
LABEL_107:
        v69 = (__int64 *)v66[2];
        result = (__int64)(v66 + 4);
        if ( v69 != v66 + 4 )
          result = _libc_free(v69, v29);
LABEL_109:
        v66 += 8;
        if ( v66 == v67 )
          goto LABEL_113;
        continue;
      }
      break;
    }
    v66 += 8;
    if ( v66 != v67 )
      continue;
    break;
  }
LABEL_113:
  if ( (v137 & 1) == 0 )
  {
    v29 = (unsigned __int64)v139 << 6;
    result = sub_C7D6A0((__int64)v138, v29, 8);
  }
LABEL_115:
  if ( v133 != v135 )
    result = _libc_free(v133, v29);
  if ( v127 != v129 )
    return _libc_free(v127, v29);
  return result;
}

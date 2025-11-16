// Function: sub_282DFE0
// Address: 0x282dfe0
//
__int64 __fastcall sub_282DFE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  __int64 v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // r8
  __int64 v8; // r9
  _QWORD *v9; // r15
  _QWORD *v10; // r14
  __int64 v11; // rax
  _QWORD *v12; // r13
  __int64 v13; // rax
  __int64 *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rbx
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // rax
  int v19; // edx
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // rax
  _QWORD *i; // rdx
  unsigned __int64 v24; // rbx
  char *v25; // rax
  void *v26; // r13
  char *v27; // r12
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  int v32; // r13d
  __int64 *v33; // r14
  unsigned __int64 v34; // rsi
  int v35; // r12d
  __int64 v36; // rax
  __int64 v37; // r13
  unsigned __int8 *v38; // rbx
  unsigned __int8 **v39; // rax
  unsigned __int8 **v40; // rax
  __int64 v41; // rax
  __int64 v42; // r15
  __int64 v43; // rax
  int v44; // edi
  int v45; // edi
  __int64 *v46; // rsi
  __int64 v47; // r11
  __int64 *v48; // rax
  __int64 v49; // r11
  _QWORD *v50; // rax
  __int64 v51; // rax
  __int64 v52; // r12
  __int64 v53; // r13
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rdx
  unsigned int v57; // eax
  _QWORD *v58; // rax
  __int64 v59; // rdx
  _QWORD *v60; // rax
  int v61; // edx
  __int64 v62; // rsi
  int v63; // edx
  unsigned int v64; // edi
  unsigned __int8 **v65; // rax
  unsigned __int8 *v66; // r8
  __int64 v67; // rdi
  unsigned int v68; // r8d
  __int64 *v69; // rax
  __int64 v70; // r10
  __int64 v71; // rsi
  int v72; // eax
  unsigned int v73; // eax
  __int64 v74; // rdx
  unsigned int v75; // eax
  __int64 v76; // r12
  __int64 *v77; // rbx
  __int64 v78; // rax
  __int64 *v79; // rax
  __int64 v80; // rsi
  _QWORD *v81; // rax
  __int64 v82; // rdx
  _QWORD *v83; // rax
  __int64 v84; // rdx
  __m128i *v85; // rbx
  int v86; // eax
  unsigned __int64 *v87; // rdi
  __int64 v88; // rax
  unsigned __int8 *v89; // rax
  __int64 v90; // rdx
  __m128i *v91; // rbx
  int v92; // eax
  unsigned __int64 *v93; // rdi
  __int64 v94; // rax
  int v95; // eax
  int v96; // esi
  int v97; // eax
  int v98; // ecx
  int v99; // eax
  int v100; // ecx
  char *v101; // rbx
  __int64 v102; // rdx
  __int64 v103; // rcx
  __int64 v104; // r8
  __int64 v105; // r9
  __int64 v106; // rbx
  __int64 *v107; // r12
  __int64 v108; // rax
  void *v109; // r14
  void *v110; // r12
  __int64 v111; // rdx
  void **v112; // rcx
  __int64 v113; // r8
  __int64 v114; // r9
  void **v115; // rax
  void **v116; // rcx
  void **v117; // rdi
  int v118; // eax
  void **v119; // rdx
  __int64 v120; // rdx
  char *v121; // rbx
  _QWORD *v123; // rbx
  _QWORD *v124; // r12
  __int64 v125; // rax
  _QWORD *v126; // rbx
  _QWORD *v127; // r12
  __int64 v128; // rax
  __int64 **v129; // rdi
  __int64 *v130; // rax
  void **v131; // rax
  __int64 **v132; // rdi
  __int64 v135; // [rsp+20h] [rbp-850h]
  __int64 v136; // [rsp+28h] [rbp-848h]
  __int64 *v137; // [rsp+30h] [rbp-840h]
  __int64 v138; // [rsp+38h] [rbp-838h]
  __int64 v139; // [rsp+40h] [rbp-830h]
  char *v140; // [rsp+48h] [rbp-828h]
  unsigned __int8 *v141; // [rsp+50h] [rbp-820h]
  __int64 v142; // [rsp+58h] [rbp-818h]
  __int64 v144; // [rsp+68h] [rbp-808h]
  __int64 v145; // [rsp+70h] [rbp-800h]
  __int64 *v146; // [rsp+88h] [rbp-7E8h]
  int v147; // [rsp+90h] [rbp-7E0h]
  int v148; // [rsp+94h] [rbp-7DCh]
  __int64 *v149; // [rsp+A0h] [rbp-7D0h]
  char *v150; // [rsp+A8h] [rbp-7C8h]
  __int64 v151; // [rsp+B0h] [rbp-7C0h] BYREF
  void *s; // [rsp+B8h] [rbp-7B8h]
  _BYTE v153[12]; // [rsp+C0h] [rbp-7B0h]
  char v154; // [rsp+CCh] [rbp-7A4h]
  char v155; // [rsp+D0h] [rbp-7A0h] BYREF
  __int64 v156[2]; // [rsp+F0h] [rbp-780h] BYREF
  _QWORD *v157; // [rsp+100h] [rbp-770h]
  __int64 v158; // [rsp+108h] [rbp-768h]
  unsigned int v159; // [rsp+110h] [rbp-760h]
  void *src; // [rsp+118h] [rbp-758h]
  char *v161; // [rsp+120h] [rbp-750h]
  char *v162; // [rsp+128h] [rbp-748h]
  __m128i v163; // [rsp+130h] [rbp-740h] BYREF
  unsigned __int8 *v164; // [rsp+140h] [rbp-730h]
  __int64 v165; // [rsp+148h] [rbp-728h]
  __int64 v166; // [rsp+150h] [rbp-720h]
  __int64 v167; // [rsp+158h] [rbp-718h]
  __int64 v168; // [rsp+160h] [rbp-710h]
  __int64 v169; // [rsp+168h] [rbp-708h]
  __int16 v170; // [rsp+170h] [rbp-700h]
  __int64 v171; // [rsp+180h] [rbp-6F0h] BYREF
  char *v172; // [rsp+188h] [rbp-6E8h]
  __int64 v173; // [rsp+190h] [rbp-6E0h]
  int v174; // [rsp+198h] [rbp-6D8h]
  char v175; // [rsp+19Ch] [rbp-6D4h]
  char v176; // [rsp+1A0h] [rbp-6D0h] BYREF
  __int64 v177; // [rsp+1E0h] [rbp-690h] BYREF
  char *v178; // [rsp+1E8h] [rbp-688h]
  __int64 v179; // [rsp+1F0h] [rbp-680h]
  int v180; // [rsp+1F8h] [rbp-678h]
  char v181; // [rsp+1FCh] [rbp-674h]
  char v182; // [rsp+200h] [rbp-670h] BYREF
  __int64 *v183; // [rsp+240h] [rbp-630h] BYREF
  unsigned __int64 v184; // [rsp+248h] [rbp-628h]
  __int64 v185; // [rsp+250h] [rbp-620h] BYREF
  _BYTE v186[4]; // [rsp+258h] [rbp-618h] BYREF
  char v187; // [rsp+25Ch] [rbp-614h]
  char v188[16]; // [rsp+260h] [rbp-610h] BYREF
  __int64 v189; // [rsp+270h] [rbp-600h] BYREF
  void **v190; // [rsp+278h] [rbp-5F8h]
  int v191; // [rsp+284h] [rbp-5ECh]
  unsigned int v192; // [rsp+288h] [rbp-5E8h]
  char v193; // [rsp+28Ch] [rbp-5E4h]
  char v194[328]; // [rsp+290h] [rbp-5E0h] BYREF
  __int64 v195; // [rsp+3D8h] [rbp-498h] BYREF
  _BYTE *v196; // [rsp+3E0h] [rbp-490h]
  __int64 v197; // [rsp+3E8h] [rbp-488h]
  int v198; // [rsp+3F0h] [rbp-480h]
  char v199; // [rsp+3F4h] [rbp-47Ch]
  _BYTE v200[64]; // [rsp+3F8h] [rbp-478h] BYREF
  _BYTE *v201; // [rsp+438h] [rbp-438h] BYREF
  __int64 v202; // [rsp+440h] [rbp-430h]
  _BYTE v203[200]; // [rsp+448h] [rbp-428h] BYREF
  int v204; // [rsp+510h] [rbp-360h] BYREF
  __int64 v205; // [rsp+518h] [rbp-358h]
  int *v206; // [rsp+520h] [rbp-350h]
  int *v207; // [rsp+528h] [rbp-348h]
  __int64 v208; // [rsp+530h] [rbp-340h]
  _QWORD v209[102]; // [rsp+540h] [rbp-330h] BYREF

  v5 = a5[9];
  memset(v209, 0, 0x300u);
  v137 = (__int64 *)v5;
  if ( v5 )
  {
    v183 = (__int64 *)v5;
    v209[0] = v5;
    v209[1] = &v209[3];
    v185 = 0x1000000000LL;
    v196 = v200;
    v209[2] = 0x1000000000LL;
    v184 = (unsigned __int64)v186;
    v195 = 0;
    v197 = 8;
    v198 = 0;
    v199 = 1;
    v201 = v203;
    v202 = 0x800000000LL;
    v204 = 0;
    v205 = 0;
    v206 = &v204;
    v207 = &v204;
    v208 = 0;
    sub_C8CF70((__int64)&v209[51], &v209[55], 8, (__int64)v200, (__int64)&v195);
    v209[64] = 0x800000000LL;
    v209[63] = &v209[65];
    if ( (_DWORD)v202 )
      sub_282DB20((__int64)&v209[63], (__int64)&v201, v6, (unsigned int)v202, v7, v8);
    if ( v205 )
    {
      v209[91] = v205;
      LODWORD(v209[90]) = v204;
      v209[92] = v206;
      v209[93] = v207;
      *(_QWORD *)(v205 + 8) = &v209[90];
      v205 = 0;
      v209[94] = v208;
      v206 = &v204;
      v207 = &v204;
      v208 = 0;
    }
    else
    {
      LODWORD(v209[90]) = 0;
      v209[91] = 0;
      v209[92] = &v209[90];
      v209[93] = &v209[90];
      v209[94] = 0;
    }
    LOBYTE(v209[95]) = 1;
    sub_282DAB0(0);
    v9 = v201;
    v10 = &v201[24 * (unsigned int)v202];
    if ( v201 != (_BYTE *)v10 )
    {
      do
      {
        v11 = *(v10 - 1);
        v10 -= 3;
        if ( v11 != 0 && v11 != -4096 && v11 != -8192 )
          sub_BD60C0(v10);
      }
      while ( v9 != v10 );
      v10 = v201;
    }
    if ( v10 != (_QWORD *)v203 )
      _libc_free((unsigned __int64)v10);
    if ( !v199 )
      _libc_free((unsigned __int64)v196);
    v12 = (_QWORD *)(v184 + 24LL * (unsigned int)v185);
    if ( (_QWORD *)v184 != v12 )
    {
      do
      {
        v13 = *(v12 - 1);
        v12 -= 3;
        if ( v13 != -4096 && v13 != 0 && v13 != -8192 )
          sub_BD60C0(v12);
      }
      while ( (_QWORD *)v184 != v12 );
      v12 = (_QWORD *)v184;
    }
    if ( v12 != (_QWORD *)v186 )
      _libc_free((unsigned __int64)v12);
    if ( byte_4F8F8E8[0] )
      nullsub_390();
    v14 = 0;
    if ( LOBYTE(v209[95]) )
      v14 = v209;
    v137 = v14;
  }
  v146 = (__int64 *)a5[5];
  v136 = a5[1];
  v138 = a5[3];
  v145 = a5[2];
  v15 = sub_AA4E30(**(_QWORD **)(a3 + 32));
  v156[0] = a3;
  v135 = v15;
  v172 = &v176;
  v178 = &v182;
  s = &v155;
  v183 = &v185;
  v184 = 0x800000000LL;
  v16 = *(_QWORD *)(v156[0] + 40) - *(_QWORD *)(v156[0] + 32);
  v171 = 0;
  v175 = 1;
  v17 = (unsigned int)(v16 >> 3);
  v173 = 8;
  v174 = 0;
  v177 = 0;
  v181 = 1;
  v179 = 8;
  v180 = 0;
  v151 = 0;
  v154 = 1;
  *(_QWORD *)v153 = 4;
  *(_DWORD *)&v153[8] = 0;
  v156[1] = 0;
  v18 = ((((((((v17 | (v17 >> 1)) >> 2) | v17 | (v17 >> 1)) >> 4) | ((v17 | (v17 >> 1)) >> 2) | v17 | (v17 >> 1)) >> 8)
        | ((((v17 | (v17 >> 1)) >> 2) | v17 | (v17 >> 1)) >> 4)
        | ((v17 | (v17 >> 1)) >> 2)
        | v17
        | (v17 >> 1)) >> 16)
      | ((((((v17 | (v17 >> 1)) >> 2) | v17 | (v17 >> 1)) >> 4) | ((v17 | (v17 >> 1)) >> 2) | v17 | (v17 >> 1)) >> 8)
      | ((((v17 | (v17 >> 1)) >> 2) | v17 | (v17 >> 1)) >> 4)
      | ((v17 | (v17 >> 1)) >> 2)
      | v17
      | (v17 >> 1);
  if ( (_DWORD)v18 == -1 )
  {
    v157 = 0;
    v158 = 0;
    v159 = 0;
  }
  else
  {
    v19 = v18 + 1;
    v20 = (((((((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
            | (4 * v19 / 3u + 1)
            | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 4)
          | (((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
          | (4 * v19 / 3u + 1)
          | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 8)
        | (((((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
          | (4 * v19 / 3u + 1)
          | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 4)
        | (((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
        | (4 * v19 / 3u + 1)
        | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1);
    v21 = ((v20 >> 16) | v20) + 1;
    v159 = v21;
    v22 = (_QWORD *)sub_C7D670(16 * v21, 8);
    v158 = 0;
    v157 = v22;
    for ( i = &v22[2 * v159]; i != v22; v22 += 2 )
    {
      if ( v22 )
        *v22 = -4096;
    }
    v17 = (unsigned int)((__int64)(*(_QWORD *)(a3 + 40) - *(_QWORD *)(a3 + 32)) >> 3);
  }
  src = 0;
  v161 = 0;
  v162 = 0;
  if ( !v17 )
    goto LABEL_40;
  v24 = 8 * v17;
  v25 = (char *)sub_22077B0(v24);
  v26 = src;
  v27 = v25;
  if ( v161 - (_BYTE *)src > 0 )
  {
    memmove(v25, src, v161 - (_BYTE *)src);
LABEL_248:
    j_j___libc_free_0((unsigned __int64)v26);
    goto LABEL_39;
  }
  if ( src )
    goto LABEL_248;
LABEL_39:
  src = v27;
  v161 = v27;
  v162 = &v27[v24];
LABEL_40:
  sub_D4E470(v156, v138);
  v139 = 0;
  if ( v137 )
    v139 = *v137;
  v32 = 0;
  v149 = &v177;
  v33 = &v171;
  while ( 1 )
  {
    if ( v137 && byte_4F8F8E8[0] )
      nullsub_390();
    v34 = (unsigned __int64)src;
    v150 = v161;
    v140 = (char *)src;
    if ( v161 != src )
    {
      v35 = v32;
      do
      {
        v36 = *((_QWORD *)v150 - 1);
        v37 = *(_QWORD *)(v36 + 56);
        v144 = v36 + 48;
        if ( v37 != v36 + 48 )
        {
          while ( 1 )
          {
            if ( !v37 )
              BUG();
            v38 = (unsigned __int8 *)(v37 - 24);
            if ( *(_BYTE *)(v37 - 24) == 84 )
            {
              if ( v154 )
              {
                v39 = (unsigned __int8 **)s;
                v28 = *(unsigned int *)&v153[4];
                v29 = (__int64)s + 8 * *(unsigned int *)&v153[4];
                if ( s != (void *)v29 )
                {
                  while ( v38 != *v39 )
                  {
                    if ( (unsigned __int8 **)v29 == ++v39 )
                      goto LABEL_148;
                  }
                  goto LABEL_56;
                }
LABEL_148:
                if ( *(_DWORD *)&v153[4] < *(_DWORD *)v153 )
                {
                  ++*(_DWORD *)&v153[4];
                  *(_QWORD *)v29 = v38;
                  ++v151;
                  if ( !*(_QWORD *)(v37 - 8) )
                    goto LABEL_150;
                  goto LABEL_57;
                }
              }
              v34 = v37 - 24;
              sub_C8CC70((__int64)&v151, v37 - 24, v28, v29, v30, v31);
            }
LABEL_56:
            if ( *(_QWORD *)(v37 - 8) )
            {
LABEL_57:
              v148 = *((_DWORD *)v33 + 5);
              v147 = *((_DWORD *)v33 + 6);
              if ( v148 != v147 )
              {
                if ( *((_BYTE *)v33 + 28) )
                {
                  v40 = (unsigned __int8 **)v33[1];
                  v28 = (__int64)&v40[v148];
                  if ( v40 == (unsigned __int8 **)v28 )
                    goto LABEL_108;
                  while ( v38 != *v40 )
                  {
                    if ( (unsigned __int8 **)v28 == ++v40 )
                      goto LABEL_108;
                  }
                }
                else
                {
                  v34 = v37 - 24;
                  if ( !sub_C8CA60((__int64)v33, v37 - 24) )
                    goto LABEL_108;
                }
              }
              v164 = 0;
              v168 = 0;
              v163.m128i_i64[0] = v135;
              v169 = 0;
              v163.m128i_i64[1] = (__int64)v146;
              v170 = 257;
              v165 = v145;
              v167 = v37 - 24;
              v166 = v136;
              v34 = (unsigned __int64)&v163;
              v41 = sub_1020E10(v37 - 24, &v163, 257, v29, v30, v31);
              v42 = v41;
              if ( v41 )
              {
                if ( *(_BYTE *)v41 > 0x1Cu )
                {
                  v43 = *(_QWORD *)(v41 + 40);
                  v28 = *(_QWORD *)(v37 + 16);
                  if ( v43 != v28 )
                  {
                    v44 = *(_DWORD *)(v138 + 24);
                    v31 = *(_QWORD *)(v138 + 8);
                    if ( v44 )
                    {
                      v45 = v44 - 1;
                      v30 = v45 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
                      v46 = (__int64 *)(v31 + 16 * v30);
                      v47 = *v46;
                      if ( v43 == *v46 )
                      {
LABEL_68:
                        v34 = v46[1];
                        if ( v34 )
                        {
                          v30 = v45 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
                          v48 = (__int64 *)(v31 + 16 * v30);
                          v49 = *v48;
                          if ( v28 != *v48 )
                          {
                            v95 = 1;
                            while ( v49 != -4096 )
                            {
                              v29 = (unsigned int)(v95 + 1);
                              v30 = v45 & (unsigned int)(v95 + v30);
                              v48 = (__int64 *)(v31 + 16LL * (unsigned int)v30);
                              v49 = *v48;
                              if ( v28 == *v48 )
                                goto LABEL_70;
                              v95 = v29;
                            }
                            goto LABEL_108;
                          }
LABEL_70:
                          v50 = (_QWORD *)v48[1];
                          if ( (_QWORD *)v34 != v50 )
                          {
                            while ( v50 )
                            {
                              v50 = (_QWORD *)*v50;
                              if ( (_QWORD *)v34 == v50 )
                                goto LABEL_73;
                            }
                            goto LABEL_108;
                          }
                        }
                      }
                      else
                      {
                        v96 = 1;
                        while ( v47 != -4096 )
                        {
                          v29 = (unsigned int)(v96 + 1);
                          v30 = v45 & (unsigned int)(v96 + v30);
                          v46 = (__int64 *)(v31 + 16LL * (unsigned int)v30);
                          v47 = *v46;
                          if ( v43 == *v46 )
                            goto LABEL_68;
                          v96 = v29;
                        }
                      }
                    }
                  }
                }
LABEL_73:
                v51 = *(_QWORD *)(v37 - 8);
                if ( v51 )
                {
                  v142 = v37;
                  v141 = (unsigned __int8 *)(v37 - 24);
                  while ( 1 )
                  {
                    v52 = *(_QWORD *)(v51 + 8);
                    v53 = *(_QWORD *)(v51 + 24);
                    if ( *(_QWORD *)v51 )
                    {
                      **(_QWORD **)(v51 + 16) = v52;
                      if ( v52 )
                        *(_QWORD *)(v52 + 16) = *(_QWORD *)(v51 + 16);
                    }
                    *(_QWORD *)v51 = v42;
                    v54 = *(_QWORD *)(v42 + 16);
                    *(_QWORD *)(v51 + 8) = v54;
                    if ( v54 )
                    {
                      v29 = v51 + 8;
                      *(_QWORD *)(v54 + 16) = v51 + 8;
                    }
                    *(_QWORD *)(v51 + 16) = v42 + 16;
                    *(_QWORD *)(v42 + 16) = v51;
                    v55 = *(_QWORD *)(v53 + 40);
                    if ( v55 )
                    {
                      v56 = (unsigned int)(*(_DWORD *)(v55 + 44) + 1);
                      v57 = *(_DWORD *)(v55 + 44) + 1;
                    }
                    else
                    {
                      v56 = 0;
                      v57 = 0;
                    }
                    if ( v57 >= *(_DWORD *)(v145 + 32) || !*(_QWORD *)(*(_QWORD *)(v145 + 24) + 8 * v56) )
                      goto LABEL_95;
                    if ( *(_BYTE *)v53 == 84 )
                    {
                      if ( v154 )
                      {
                        v58 = s;
                        v59 = (__int64)s + 8 * *(unsigned int *)&v153[4];
                        if ( s != (void *)v59 )
                        {
                          while ( v53 != *v58 )
                          {
                            if ( (_QWORD *)v59 == ++v58 )
                              goto LABEL_134;
                          }
LABEL_90:
                          if ( !*((_BYTE *)v149 + 28) )
                            goto LABEL_169;
                          v60 = (_QWORD *)v149[1];
                          v59 = *((unsigned int *)v149 + 5);
                          v29 = (__int64)&v60[v59];
                          if ( v60 == (_QWORD *)v29 )
                          {
LABEL_172:
                            if ( (unsigned int)v59 >= *((_DWORD *)v149 + 4) )
                            {
LABEL_169:
                              sub_C8CC70((__int64)v149, v53, v59, v29, v30, v31);
                              goto LABEL_95;
                            }
                            *((_DWORD *)v149 + 5) = v59 + 1;
                            *(_QWORD *)v29 = v53;
                            ++*v149;
                          }
                          else
                          {
                            while ( v53 != *v60 )
                            {
                              if ( (_QWORD *)v29 == ++v60 )
                                goto LABEL_172;
                            }
                          }
                          goto LABEL_95;
                        }
                      }
                      else if ( sub_C8CA60((__int64)&v151, v53) )
                      {
                        goto LABEL_90;
                      }
                    }
LABEL_134:
                    if ( v148 == v147 )
                      goto LABEL_95;
                    v29 = a3;
                    v80 = *(_QWORD *)(v53 + 40);
                    if ( *(_BYTE *)(a3 + 84) )
                    {
                      v81 = *(_QWORD **)(a3 + 64);
                      v82 = (__int64)&v81[*(unsigned int *)(a3 + 76)];
                      if ( v81 == (_QWORD *)v82 )
                        goto LABEL_95;
                      while ( v80 != *v81 )
                      {
                        if ( (_QWORD *)v82 == ++v81 )
                          goto LABEL_95;
                      }
                      if ( !*((_BYTE *)v33 + 28) )
                        goto LABEL_166;
                    }
                    else
                    {
                      if ( !sub_C8CA60(a3 + 56, v80) )
                        goto LABEL_95;
                      if ( !*((_BYTE *)v33 + 28) )
                      {
LABEL_166:
                        sub_C8CC70((__int64)v33, v53, v82, v29, v30, v31);
                        goto LABEL_95;
                      }
                    }
                    v83 = (_QWORD *)v33[1];
                    v82 = *((unsigned int *)v33 + 5);
                    v29 = (__int64)&v83[v82];
                    if ( v83 == (_QWORD *)v29 )
                    {
LABEL_170:
                      if ( (unsigned int)v82 >= *((_DWORD *)v33 + 4) )
                        goto LABEL_166;
                      *((_DWORD *)v33 + 5) = v82 + 1;
                      *(_QWORD *)v29 = v53;
                      ++*v33;
                    }
                    else
                    {
                      while ( v53 != *v83 )
                      {
                        if ( (_QWORD *)v29 == ++v83 )
                          goto LABEL_170;
                      }
                    }
LABEL_95:
                    if ( !v52 )
                    {
                      v37 = v142;
                      v38 = v141;
                      break;
                    }
                    v51 = v52;
                  }
                }
                if ( v137 )
                {
                  if ( *(_BYTE *)v42 > 0x1Cu )
                  {
                    v61 = *(_DWORD *)(v139 + 56);
                    v62 = *(_QWORD *)(v139 + 40);
                    if ( v61 )
                    {
                      v63 = v61 - 1;
                      v64 = v63 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
                      v65 = (unsigned __int8 **)(v62 + 16LL * v64);
                      v66 = *v65;
                      if ( v38 == *v65 )
                      {
LABEL_102:
                        v67 = (__int64)v65[1];
                        if ( v67 )
                        {
                          v68 = v63 & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
                          v69 = (__int64 *)(v62 + 16LL * v68);
                          v70 = *v69;
                          if ( v42 == *v69 )
                          {
LABEL_104:
                            v71 = v69[1];
                            if ( v71 )
                              sub_BD84D0(v67, v71);
                          }
                          else
                          {
                            v99 = 1;
                            while ( v70 != -4096 )
                            {
                              v100 = v99 + 1;
                              v68 = v63 & (v99 + v68);
                              v69 = (__int64 *)(v62 + 16LL * v68);
                              v70 = *v69;
                              if ( v42 == *v69 )
                                goto LABEL_104;
                              v99 = v100;
                            }
                          }
                        }
                      }
                      else
                      {
                        v97 = 1;
                        while ( v66 != (unsigned __int8 *)-4096LL )
                        {
                          v98 = v97 + 1;
                          v64 = v63 & (v97 + v64);
                          v65 = (unsigned __int8 **)(v62 + 16LL * v64);
                          v66 = *v65;
                          if ( v38 == *v65 )
                            goto LABEL_102;
                          v97 = v98;
                        }
                      }
                    }
                  }
                }
                v34 = (unsigned __int64)v146;
                LOBYTE(v72) = sub_F50EE0(v38, v146);
                v35 = v72;
                if ( !(_BYTE)v72 )
                {
                  v35 = 1;
                  goto LABEL_108;
                }
                v163 = (__m128i)6uLL;
                v164 = v38;
                if ( v38 != (unsigned __int8 *)-8192LL && v38 != (unsigned __int8 *)-4096LL )
                  sub_BD73F0((__int64)&v163);
                v90 = (unsigned int)v184;
                v29 = (__int64)v183;
                v91 = &v163;
                v34 = (unsigned int)v184 + 1LL;
                v92 = v184;
                if ( v34 > HIDWORD(v184) )
                {
                  if ( v183 > (__int64 *)&v163 || &v163 >= (__m128i *)&v183[3 * (unsigned int)v184] )
                  {
                    sub_F39130((__int64)&v183, v34, (unsigned int)v184, (__int64)v183, v30, v31);
                    v90 = (unsigned int)v184;
                    v29 = (__int64)v183;
                    v91 = &v163;
                    v92 = v184;
                  }
                  else
                  {
                    v101 = (char *)((char *)&v163 - (char *)v183);
                    sub_F39130((__int64)&v183, v34, (unsigned int)v184, (__int64)v183, v30, v31);
                    v29 = (__int64)v183;
                    v90 = (unsigned int)v184;
                    v91 = (__m128i *)&v101[(_QWORD)v183];
                    v92 = v184;
                  }
                }
                v28 = 3 * v90;
                v93 = (unsigned __int64 *)(v29 + 8 * v28);
                if ( v93 )
                {
                  *v93 = 6;
                  v94 = v91[1].m128i_i64[0];
                  v93[1] = 0;
                  v93[2] = v94;
                  if ( v94 != -4096 && v94 != 0 && v94 != -8192 )
                  {
                    v34 = v91->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
                    sub_BD6050(v93, v34);
                  }
                  v92 = v184;
                }
                LODWORD(v184) = v92 + 1;
                v89 = v164;
                LOBYTE(v29) = v164 + 4096 != 0;
                LOBYTE(v28) = v164 != 0;
                if ( ((v164 != 0) & (unsigned __int8)v29) != 0 )
                  goto LABEL_161;
                v37 = *(_QWORD *)(v37 + 8);
                if ( v144 == v37 )
                  break;
              }
              else
              {
LABEL_108:
                v37 = *(_QWORD *)(v37 + 8);
                if ( v144 == v37 )
                  break;
              }
            }
            else
            {
LABEL_150:
              v34 = (unsigned __int64)v146;
              if ( !sub_F50EE0((unsigned __int8 *)(v37 - 24), v146) )
                goto LABEL_108;
              v164 = (unsigned __int8 *)(v37 - 24);
              v163 = (__m128i)6uLL;
              if ( v37 != -8168 && v37 != -4072 )
                sub_BD73F0((__int64)&v163);
              v84 = (unsigned int)v184;
              v29 = (__int64)v183;
              v85 = &v163;
              v34 = (unsigned int)v184 + 1LL;
              v86 = v184;
              if ( v34 > HIDWORD(v184) )
              {
                if ( v183 > (__int64 *)&v163 || &v163 >= (__m128i *)&v183[3 * (unsigned int)v184] )
                {
                  sub_F39130((__int64)&v183, v34, (unsigned int)v184, (__int64)v183, v30, v31);
                  v84 = (unsigned int)v184;
                  v29 = (__int64)v183;
                  v86 = v184;
                }
                else
                {
                  v121 = (char *)((char *)&v163 - (char *)v183);
                  sub_F39130((__int64)&v183, v34, (unsigned int)v184, (__int64)v183, v30, v31);
                  v29 = (__int64)v183;
                  v84 = (unsigned int)v184;
                  v85 = (__m128i *)&v121[(_QWORD)v183];
                  v86 = v184;
                }
              }
              v28 = 3 * v84;
              v87 = (unsigned __int64 *)(v29 + 8 * v28);
              if ( v87 )
              {
                *v87 = 6;
                v88 = v85[1].m128i_i64[0];
                v87[1] = 0;
                v87[2] = v88;
                if ( v88 != -4096 && v88 != 0 && v88 != -8192 )
                {
                  v34 = v85->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
                  sub_BD6050(v87, v34);
                }
                v86 = v184;
              }
              LODWORD(v184) = v86 + 1;
              v89 = v164;
              LOBYTE(v29) = v164 != 0;
              LOBYTE(v28) = v164 + 4096 != 0;
              if ( ((unsigned __int8)v28 & (v164 != 0)) == 0 )
                goto LABEL_108;
LABEL_161:
              if ( v89 == (unsigned __int8 *)-8192LL )
                goto LABEL_108;
              sub_BD60C0(&v163);
              v37 = *(_QWORD *)(v37 + 8);
              if ( v144 == v37 )
                break;
            }
          }
        }
        v150 -= 8;
      }
      while ( v140 != v150 );
      v32 = v35;
    }
    if ( (_DWORD)v184 )
    {
      v34 = (unsigned __int64)v146;
      v164 = 0;
      sub_F5C330((__int64)&v183, v146, v137, (__int64)&v163);
      if ( v164 )
      {
        v34 = (unsigned __int64)&v163;
        ((void (__fastcall *)(__m128i *, __m128i *, __int64))v164)(&v163, &v163, 3);
      }
      v32 = 1;
    }
    if ( v137 && byte_4F8F8E8[0] )
      break;
    v29 = *((unsigned int *)v149 + 6);
    if ( *((_DWORD *)v149 + 5) == (_DWORD)v29 )
      goto LABEL_210;
LABEL_115:
    ++*v33;
    if ( *((_BYTE *)v33 + 28) )
      goto LABEL_120;
    v73 = 4 * (*((_DWORD *)v33 + 5) - *((_DWORD *)v33 + 6));
    v74 = *((unsigned int *)v33 + 4);
    if ( v73 < 0x20 )
      v73 = 32;
    if ( (unsigned int)v74 <= v73 )
    {
      v34 = 0xFFFFFFFFLL;
      memset((void *)v33[1], -1, 8 * v74);
LABEL_120:
      *(__int64 *)((char *)v33 + 20) = 0;
      goto LABEL_121;
    }
    sub_C8C990((__int64)v33, v34);
LABEL_121:
    ++v151;
    if ( !v154 )
    {
      v75 = 4 * (*(_DWORD *)&v153[4] - *(_DWORD *)&v153[8]);
      if ( v75 < 0x20 )
        v75 = 32;
      if ( *(_DWORD *)v153 > v75 )
      {
        sub_C8C990((__int64)&v151, v34);
        goto LABEL_127;
      }
      memset(s, -1, 8LL * *(unsigned int *)v153);
    }
    *(_QWORD *)&v153[4] = 0;
LABEL_127:
    v76 = (__int64)v183;
    v77 = &v183[3 * (unsigned int)v184];
    while ( (__int64 *)v76 != v77 )
    {
      while ( 1 )
      {
        v78 = *(v77 - 1);
        v77 -= 3;
        LOBYTE(v29) = v78 != 0;
        LOBYTE(v28) = v78 != -4096;
        if ( ((unsigned __int8)v28 & (v78 != 0)) == 0 || v78 == -8192 )
          break;
        sub_BD60C0(v77);
        if ( (__int64 *)v76 == v77 )
          goto LABEL_132;
      }
    }
LABEL_132:
    LODWORD(v184) = 0;
    v79 = v33;
    v33 = v149;
    v149 = v79;
  }
  v34 = 0;
  nullsub_390();
  v29 = *((unsigned int *)v149 + 6);
  if ( *((_DWORD *)v149 + 5) != (_DWORD)v29 )
    goto LABEL_115;
LABEL_210:
  if ( src )
    j_j___libc_free_0((unsigned __int64)src);
  sub_C7D6A0((__int64)v157, 16LL * v159, 8);
  v106 = (__int64)v183;
  v107 = &v183[3 * (unsigned int)v184];
  if ( v183 != v107 )
  {
    do
    {
      v108 = *(v107 - 1);
      v107 -= 3;
      LOBYTE(v103) = v108 != -4096;
      LOBYTE(v102) = v108 != 0;
      if ( ((v108 != 0) & (unsigned __int8)v103) != 0 && v108 != -8192 )
        sub_BD60C0(v107);
    }
    while ( (__int64 *)v106 != v107 );
    v107 = v183;
  }
  if ( v107 != &v185 )
    _libc_free((unsigned __int64)v107);
  if ( !v154 )
    _libc_free((unsigned __int64)s);
  if ( v181 )
  {
    if ( v175 )
      goto LABEL_224;
LABEL_250:
    _libc_free((unsigned __int64)v172);
  }
  else
  {
    _libc_free((unsigned __int64)v178);
    if ( !v175 )
      goto LABEL_250;
  }
LABEL_224:
  v109 = (void *)(a1 + 32);
  v110 = (void *)(a1 + 80);
  if ( !(_BYTE)v32 )
  {
    *(_QWORD *)(a1 + 8) = v109;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v110;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_245;
  }
  sub_22D0390((__int64)&v183, a1, v102, v103, v104, v105);
  if ( v191 != v192 )
    goto LABEL_226;
  if ( !v187 )
  {
    if ( sub_C8CA60((__int64)&v183, (__int64)&qword_4F82400) )
      goto LABEL_231;
LABEL_226:
    if ( !v187 )
    {
LABEL_277:
      sub_C8CC70((__int64)&v183, (__int64)&unk_4F82408, v111, (__int64)v112, v113, v114);
      goto LABEL_231;
    }
    v115 = (void **)v184;
    v111 = HIDWORD(v185);
    v112 = (void **)(v184 + 8LL * HIDWORD(v185));
    if ( v112 != (void **)v184 )
      goto LABEL_230;
LABEL_275:
    if ( (unsigned int)v111 < (unsigned int)v185 )
    {
      HIDWORD(v185) = v111 + 1;
      *v112 = &unk_4F82408;
      v183 = (__int64 *)((char *)v183 + 1);
      goto LABEL_231;
    }
    goto LABEL_277;
  }
  v115 = (void **)v184;
  v111 = HIDWORD(v185);
  v112 = (void **)(v184 + 8LL * HIDWORD(v185));
  if ( (void **)v184 == v112 )
    goto LABEL_275;
  v129 = (__int64 **)v184;
  while ( *v129 != &qword_4F82400 )
  {
    if ( v112 == (void **)++v129 )
    {
LABEL_230:
      while ( *v115 != &unk_4F82408 )
      {
        if ( ++v115 == v112 )
          goto LABEL_275;
      }
      break;
    }
  }
LABEL_231:
  if ( a5[9] )
  {
    if ( v193 )
    {
      v116 = v190;
      v117 = &v190[v191];
      v118 = v191;
      if ( v190 == v117 )
      {
LABEL_294:
        v120 = v192;
      }
      else
      {
        v119 = v190;
        while ( *v119 != &unk_4F8F810 )
        {
          if ( v117 == ++v119 )
            goto LABEL_294;
        }
        *v119 = v190[--v191];
        v118 = v191;
        ++v189;
        v120 = v192;
      }
    }
    else
    {
      v130 = sub_C8CA60((__int64)&v189, (__int64)&unk_4F8F810);
      if ( v130 )
      {
        *v130 = -2;
        ++v189;
        v120 = v192 + 1;
        v118 = v191;
        ++v192;
      }
      else
      {
        v118 = v191;
        v120 = v192;
      }
    }
    if ( (_DWORD)v120 == v118 )
    {
      if ( v187 )
      {
        v131 = (void **)v184;
        v120 = HIDWORD(v185);
        v116 = (void **)(v184 + 8LL * HIDWORD(v185));
        if ( (void **)v184 == v116 )
          goto LABEL_295;
        v132 = (__int64 **)v184;
        while ( *v132 != &qword_4F82400 )
        {
          if ( v116 == (void **)++v132 )
          {
LABEL_291:
            while ( *v131 != &unk_4F8F810 )
            {
              if ( ++v131 == v116 )
                goto LABEL_295;
            }
            break;
          }
        }
      }
      else if ( !sub_C8CA60((__int64)&v183, (__int64)&qword_4F82400) )
      {
        goto LABEL_287;
      }
    }
    else
    {
LABEL_287:
      if ( !v187 )
        goto LABEL_269;
      v131 = (void **)v184;
      v120 = HIDWORD(v185);
      v116 = (void **)(v184 + 8LL * HIDWORD(v185));
      if ( (void **)v184 != v116 )
        goto LABEL_291;
LABEL_295:
      if ( (unsigned int)v120 < (unsigned int)v185 )
      {
        HIDWORD(v185) = v120 + 1;
        *v116 = &unk_4F8F810;
        v183 = (__int64 *)((char *)v183 + 1);
      }
      else
      {
LABEL_269:
        sub_C8CC70((__int64)&v183, (__int64)&unk_4F8F810, v120, (__int64)v116, v113, v114);
      }
    }
  }
  sub_C8CF70(a1, v109, 2, (__int64)v188, (__int64)&v183);
  sub_C8CF70(a1 + 48, v110, 2, (__int64)v194, (__int64)&v189);
  if ( !v193 )
    _libc_free((unsigned __int64)v190);
  if ( !v187 )
    _libc_free(v184);
LABEL_245:
  if ( LOBYTE(v209[95]) )
  {
    LOBYTE(v209[95]) = 0;
    sub_282DAB0((_QWORD *)v209[91]);
    v123 = (_QWORD *)v209[63];
    v124 = (_QWORD *)(v209[63] + 24LL * LODWORD(v209[64]));
    if ( (_QWORD *)v209[63] != v124 )
    {
      do
      {
        v125 = *(v124 - 1);
        v124 -= 3;
        if ( v125 != 0 && v125 != -4096 && v125 != -8192 )
          sub_BD60C0(v124);
      }
      while ( v123 != v124 );
      v124 = (_QWORD *)v209[63];
    }
    if ( v124 != &v209[65] )
      _libc_free((unsigned __int64)v124);
    if ( !BYTE4(v209[54]) )
      _libc_free(v209[52]);
    v126 = (_QWORD *)v209[1];
    v127 = (_QWORD *)(v209[1] + 24LL * LODWORD(v209[2]));
    if ( (_QWORD *)v209[1] != v127 )
    {
      do
      {
        v128 = *(v127 - 1);
        v127 -= 3;
        if ( v128 != 0 && v128 != -4096 && v128 != -8192 )
          sub_BD60C0(v127);
      }
      while ( v126 != v127 );
      v127 = (_QWORD *)v209[1];
    }
    if ( v127 != &v209[3] )
      _libc_free((unsigned __int64)v127);
  }
  return a1;
}

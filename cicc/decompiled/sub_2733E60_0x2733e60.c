// Function: sub_2733E60
// Address: 0x2733e60
//
__int64 __fastcall sub_2733E60(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // rax
  int v11; // edx
  __int64 v12; // rcx
  int v13; // eax
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 *v16; // rdi
  int v18; // eax
  __int64 *v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rbx
  __int64 *v22; // r12
  __m128i *v23; // r13
  __int64 v24; // r15
  __int64 v25; // rbx
  __int64 v26; // rdx
  unsigned int v27; // eax
  char v28; // di
  unsigned int v29; // eax
  __int64 *v30; // rsi
  _QWORD *v31; // rax
  _QWORD *v32; // rax
  __int64 *v33; // rax
  __int64 v34; // rsi
  __int64 *v35; // rbx
  __int64 v36; // r12
  __int64 *v37; // r13
  _QWORD *v38; // rax
  __int64 *v39; // rax
  _QWORD *v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rax
  unsigned int v43; // r12d
  __int64 v44; // rdx
  unsigned int v45; // eax
  __int64 v46; // rax
  __int64 v47; // r9
  __int64 **v48; // r14
  __int64 **v49; // r15
  __int64 *v50; // r13
  __int64 v51; // rsi
  __int64 *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // r13
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  unsigned int v57; // eax
  int v58; // eax
  int v59; // edi
  unsigned int v60; // r8d
  unsigned __int64 v61; // rdx
  __int64 v62; // r13
  _QWORD *v63; // r15
  int v64; // r10d
  __int64 *v65; // rax
  __int64 v66; // rdx
  __int64 *v67; // r12
  __int64 v68; // rcx
  __int64 v69; // rdx
  unsigned int v70; // eax
  __int64 v71; // r13
  int v72; // r12d
  _QWORD *v73; // rdx
  unsigned int v74; // edi
  _QWORD *v75; // rax
  __int64 v76; // r8
  __int64 v77; // r12
  __int64 v78; // rax
  unsigned __int64 v79; // rax
  __int64 v80; // rdi
  __int64 v81; // rax
  bool v82; // cf
  __int64 v83; // rax
  __int64 v84; // rsi
  int v85; // r14d
  int v86; // r9d
  int v87; // eax
  __int64 v88; // rcx
  int v89; // edx
  __int64 v90; // rcx
  __int64 v91; // r9
  __int64 v92; // rdx
  _QWORD *v93; // rax
  _QWORD *v94; // rdx
  __int64 v95; // rax
  __int64 *v96; // r12
  __int64 *v97; // rbx
  __int64 *v98; // rsi
  unsigned __int64 v99; // rbx
  unsigned __int64 v100; // rdi
  __int64 *v101; // r13
  __int64 *v102; // r12
  __int8 v103; // cl
  __int8 v104; // dl
  __int64 v105; // rax
  __int16 v106; // dx
  __int64 v107; // rax
  __int64 *v108; // r14
  __int64 *v109; // r13
  __int64 *v110; // rsi
  __int64 v111; // rax
  unsigned __int8 *v112; // rsi
  __int64 v113; // rdx
  __int64 v114; // r9
  __int64 v115; // rcx
  int v116; // edx
  unsigned int v117; // esi
  __int64 *v118; // r10
  __int64 v119; // r11
  __int64 v120; // rax
  __int64 v121; // rsi
  unsigned int v122; // edi
  __int64 *v123; // r8
  __int64 v124; // r10
  __int64 v125; // rax
  __int64 v126; // rdi
  __int64 v127; // rax
  unsigned int v128; // r8d
  __int64 v129; // rdx
  __int64 v130; // rsi
  unsigned int v131; // ecx
  __int64 v132; // rdx
  __int64 v133; // rcx
  int v134; // r10d
  int v135; // r8d
  int v136; // r11d
  int v137; // eax
  int v138; // r9d
  __int64 *v139; // rdi
  __int64 v140; // rcx
  __int64 v141; // rcx
  __int64 v142; // r8
  int v143; // edi
  _QWORD *v144; // rsi
  int v145; // edi
  __int64 v146; // rcx
  __int64 v147; // r8
  unsigned int v148; // ecx
  _QWORD *v149; // rdi
  __int64 v150; // rsi
  unsigned int v151; // eax
  int v152; // eax
  unsigned __int64 v153; // rax
  unsigned __int64 v154; // rax
  int v155; // ebx
  __int64 v156; // r12
  _QWORD *v157; // rax
  _QWORD *i; // rdx
  int v159; // r13d
  int v160; // r8d
  __int64 *v161; // rdi
  _QWORD *v162; // rsi
  __int64 *v164; // [rsp+28h] [rbp-238h]
  __int64 v165; // [rsp+28h] [rbp-238h]
  __int64 *v166; // [rsp+30h] [rbp-230h]
  __int64 *v168; // [rsp+38h] [rbp-228h]
  int v169; // [rsp+38h] [rbp-228h]
  _QWORD *v170; // [rsp+38h] [rbp-228h]
  __int64 v171; // [rsp+40h] [rbp-220h] BYREF
  __int64 v172; // [rsp+48h] [rbp-218h] BYREF
  __int64 v173; // [rsp+50h] [rbp-210h] BYREF
  __int64 v174; // [rsp+58h] [rbp-208h]
  __int64 v175; // [rsp+60h] [rbp-200h]
  unsigned int v176; // [rsp+68h] [rbp-1F8h]
  __int64 v177; // [rsp+70h] [rbp-1F0h] BYREF
  _QWORD *v178; // [rsp+78h] [rbp-1E8h]
  __int64 v179; // [rsp+80h] [rbp-1E0h]
  __int64 v180; // [rsp+88h] [rbp-1D8h]
  __int64 *v181; // [rsp+90h] [rbp-1D0h]
  __int64 v182; // [rsp+98h] [rbp-1C8h]
  __int64 v183; // [rsp+A0h] [rbp-1C0h] BYREF
  void *s; // [rsp+A8h] [rbp-1B8h]
  _BYTE v185[12]; // [rsp+B0h] [rbp-1B0h]
  char v186; // [rsp+BCh] [rbp-1A4h]
  char v187; // [rsp+C0h] [rbp-1A0h] BYREF
  _QWORD *v188; // [rsp+100h] [rbp-160h] BYREF
  __int64 v189; // [rsp+108h] [rbp-158h]
  _QWORD v190[16]; // [rsp+110h] [rbp-150h] BYREF
  __m128i v191; // [rsp+190h] [rbp-D0h] BYREF
  __int64 v192; // [rsp+1A0h] [rbp-C0h]
  int v193; // [rsp+1A8h] [rbp-B8h]
  unsigned __int8 v194; // [rsp+1ACh] [rbp-B4h]
  char v195; // [rsp+1B0h] [rbp-B0h] BYREF

  v6 = 16 * a5;
  v7 = (__int64)a2;
  v8 = a4 + v6;
  v177 = 0;
  v178 = 0;
  v179 = 0;
  v180 = 0;
  v181 = &v183;
  v182 = 0;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0;
  if ( a4 + v6 == a4 )
  {
    v12 = a2[5];
    v166 = (__int64 *)a2[2];
    if ( v166 )
    {
      v19 = &v183;
      v20 = 0;
LABEL_15:
      v21 = *(_QWORD *)(v7 + 8);
      v171 = v12;
      s = &v187;
      v191.m128i_i64[1] = (__int64)&v195;
      v183 = 0;
      *(_QWORD *)v185 = 8;
      *(_DWORD *)&v185[8] = 0;
      v186 = 1;
      v191.m128i_i64[0] = 0;
      v192 = 16;
      v193 = 0;
      v194 = 1;
      v168 = &v19[v20];
      if ( v168 == v19 )
        goto LABEL_56;
      v22 = v19;
      v23 = &v191;
      v24 = v21;
      while ( 1 )
      {
        v25 = *v22;
        if ( *v22 )
        {
          v26 = (unsigned int)(*(_DWORD *)(v25 + 44) + 1);
          v27 = *(_DWORD *)(v25 + 44) + 1;
        }
        else
        {
          v26 = 0;
          v27 = 0;
        }
        if ( v27 >= *(_DWORD *)(v24 + 32) || !*(_QWORD *)(*(_QWORD *)(v24 + 24) + 8 * v26) )
        {
LABEL_17:
          if ( v168 == ++v22 )
            goto LABEL_55;
          continue;
        }
        v28 = v186;
        ++v183;
        if ( !v186 )
        {
          v29 = 4 * (*(_DWORD *)&v185[4] - *(_DWORD *)&v185[8]);
          if ( v29 < 0x20 )
            v29 = 32;
          if ( *(_DWORD *)v185 > v29 )
          {
            sub_C8C990((__int64)&v183, 32);
            v28 = v186;
            goto LABEL_28;
          }
          memset(s, -1, 8LL * *(unsigned int *)v185);
          v28 = v186;
        }
        *(_QWORD *)&v185[4] = 0;
LABEL_28:
        if ( !v28 )
          goto LABEL_81;
LABEL_29:
        v30 = (__int64 *)s;
        v26 = (__int64)s + 8 * *(unsigned int *)&v185[4];
        v12 = *(unsigned int *)&v185[4];
        if ( s != (void *)v26 )
        {
          v31 = s;
          while ( *v31 != v25 )
          {
            if ( (_QWORD *)v26 == ++v31 )
              goto LABEL_85;
          }
          goto LABEL_33;
        }
LABEL_85:
        if ( *(_DWORD *)&v185[4] < *(_DWORD *)v185 )
        {
          v12 = (unsigned int)++*(_DWORD *)&v185[4];
          *(_QWORD *)v26 = v25;
          v30 = (__int64 *)s;
          ++v183;
          v28 = v186;
          goto LABEL_33;
        }
        while ( 1 )
        {
LABEL_81:
          sub_C8CC70((__int64)&v183, v25, v26, v12, v6, a6);
          v28 = v186;
          v30 = (__int64 *)s;
LABEL_33:
          if ( v171 == v25 )
            goto LABEL_39;
          if ( v194 )
            break;
          if ( sub_C8CA60((__int64)v23, v25) )
          {
            v28 = v186;
            v30 = (__int64 *)s;
            goto LABEL_39;
          }
LABEL_73:
          if ( v25 )
          {
            v26 = (unsigned int)(*(_DWORD *)(v25 + 44) + 1);
            v57 = *(_DWORD *)(v25 + 44) + 1;
          }
          else
          {
            v26 = 0;
            v57 = 0;
          }
          if ( v57 >= *(_DWORD *)(v24 + 32) )
            BUG();
          v25 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v24 + 24) + 8 * v26) + 8LL);
          if ( (_DWORD)v180 )
          {
            v26 = (unsigned int)(v180 - 1);
            v58 = v26 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
            v12 = v178[v58];
            if ( v25 == v12 )
              goto LABEL_17;
            v59 = 1;
            while ( v12 != -4096 )
            {
              v6 = (unsigned int)(v59 + 1);
              v58 = v26 & (v59 + v58);
              v12 = v178[v58];
              if ( v25 == v12 )
                goto LABEL_17;
              ++v59;
            }
          }
          v28 = v186;
          if ( v186 )
            goto LABEL_29;
        }
        v32 = (_QWORD *)v191.m128i_i64[1];
        v26 = v191.m128i_i64[1] + 8LL * HIDWORD(v192);
        if ( v191.m128i_i64[1] == v26 )
          goto LABEL_73;
        while ( *v32 != v25 )
        {
          if ( (_QWORD *)v26 == ++v32 )
            goto LABEL_73;
        }
LABEL_39:
        if ( v28 )
          v6 = (__int64)&v30[*(unsigned int *)&v185[4]];
        else
          v6 = (__int64)&v30[*(unsigned int *)v185];
        v33 = v30;
        if ( v30 == (__int64 *)v6 )
          goto LABEL_17;
        while ( 1 )
        {
          v34 = *v33;
          v35 = v33;
          if ( (unsigned __int64)*v33 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( (__int64 *)v6 == ++v33 )
            goto LABEL_17;
        }
        if ( (__int64 *)v6 == v33 )
          goto LABEL_17;
        a6 = v194;
        v164 = v22;
        v36 = (__int64)v23;
        v37 = (__int64 *)v6;
        if ( v194 )
        {
LABEL_47:
          v38 = (_QWORD *)v191.m128i_i64[1];
          v26 = v191.m128i_i64[1] + 8LL * HIDWORD(v192);
          if ( v191.m128i_i64[1] != v26 )
          {
            do
            {
              if ( *v38 == v34 )
                goto LABEL_51;
              ++v38;
            }
            while ( (_QWORD *)v26 != v38 );
          }
          if ( HIDWORD(v192) < (unsigned int)v192 )
          {
            ++HIDWORD(v192);
            *(_QWORD *)v26 = v34;
            a6 = v194;
            ++v191.m128i_i64[0];
            goto LABEL_51;
          }
        }
        while ( 1 )
        {
          sub_C8CC70(v36, v34, v26, v12, v6, a6);
          a6 = v194;
LABEL_51:
          v39 = v35 + 1;
          if ( v35 + 1 == v37 )
            break;
          while ( 1 )
          {
            v34 = *v39;
            v35 = v39;
            if ( (unsigned __int64)*v39 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v37 == ++v39 )
              goto LABEL_54;
          }
          if ( v37 == v39 )
            break;
          if ( (_BYTE)a6 )
            goto LABEL_47;
        }
LABEL_54:
        v23 = (__m128i *)v36;
        v22 = v164 + 1;
        if ( v168 == v164 + 1 )
        {
LABEL_55:
          v12 = v171;
          v21 = v24;
LABEL_56:
          v40 = v190;
          v190[0] = v12;
          v188 = v190;
          v169 = 0;
          v189 = 0x1000000001LL;
          v41 = 0;
          while ( 2 )
          {
            v42 = v40[v41];
            v43 = v169 + 1;
            if ( v42 )
            {
              v44 = (unsigned int)(*(_DWORD *)(v42 + 44) + 1);
              v45 = *(_DWORD *)(v42 + 44) + 1;
            }
            else
            {
              v44 = 0;
              v45 = 0;
            }
            if ( v45 >= *(_DWORD *)(v21 + 32) )
              BUG();
            v46 = *(_QWORD *)(*(_QWORD *)(v21 + 24) + 8 * v44);
            v47 = *(_QWORD *)(v46 + 24);
            v48 = (__int64 **)(v47 + 8LL * *(unsigned int *)(v46 + 32));
            v49 = (__int64 **)v47;
            if ( (__int64 **)v47 != v48 )
            {
              while ( 2 )
              {
                v50 = *v49;
                v51 = **v49;
                if ( v194 )
                {
                  v52 = (__int64 *)v191.m128i_i64[1];
                  v53 = v191.m128i_i64[1] + 8LL * HIDWORD(v192);
                  if ( v191.m128i_i64[1] == v53 )
                    goto LABEL_69;
                  while ( 1 )
                  {
                    v54 = *v52;
                    if ( v51 == *v52 )
                      break;
                    if ( (__int64 *)v53 == ++v52 )
                      goto LABEL_69;
                  }
                }
                else
                {
                  if ( !sub_C8CA60((__int64)&v191, v51) )
                    goto LABEL_69;
                  v54 = *v50;
                }
                v55 = (unsigned int)v189;
                v56 = (unsigned int)v189 + 1LL;
                if ( v56 > HIDWORD(v189) )
                {
                  sub_C8D5F0((__int64)&v188, v190, v56, 8u, v6, v47);
                  v55 = (unsigned int)v189;
                }
                v188[v55] = v54;
                LODWORD(v189) = v189 + 1;
LABEL_69:
                if ( v48 == ++v49 )
                  break;
                continue;
              }
            }
            v41 = v43;
            if ( (_DWORD)v189 != v43 )
            {
              ++v169;
              v40 = v188;
              continue;
            }
            break;
          }
          v174 = 0;
          v175 = 0;
          v176 = 0;
          v60 = v169 + 2;
          if ( v169 == -2 )
          {
            v62 = 0;
            v173 = 1;
            v63 = &v188[v43];
            v165 = (__int64)v188;
            if ( v188 != v63 )
              goto LABEL_113;
            goto LABEL_144;
          }
          v173 = 1;
          v61 = ((((((4 * v60 / 3 + 1) | ((unsigned __int64)(4 * v60 / 3 + 1) >> 1)) >> 2)
                 | (4 * v60 / 3 + 1)
                 | ((unsigned __int64)(4 * v60 / 3 + 1) >> 1)) >> 4)
               | (((4 * v60 / 3 + 1) | ((unsigned __int64)(4 * v60 / 3 + 1) >> 1)) >> 2)
               | (4 * v60 / 3 + 1)
               | ((unsigned __int64)(4 * v60 / 3 + 1) >> 1)) >> 8;
          sub_2730F10(
            (__int64)&v173,
            (((v61
             | (((((4 * v60 / 3 + 1) | ((unsigned __int64)(4 * v60 / 3 + 1) >> 1)) >> 2)
               | (4 * v60 / 3 + 1)
               | ((unsigned __int64)(4 * v60 / 3 + 1) >> 1)) >> 4)
             | (((4 * v60 / 3 + 1) | ((unsigned __int64)(4 * v60 / 3 + 1) >> 1)) >> 2)
             | (4 * v60 / 3 + 1)
             | ((unsigned __int64)(4 * v60 / 3 + 1) >> 1)) >> 16)
           | v61
           | (((((4 * v60 / 3 + 1) | ((unsigned __int64)(4 * v60 / 3 + 1) >> 1)) >> 2)
             | (4 * v60 / 3 + 1)
             | ((unsigned __int64)(4 * v60 / 3 + 1) >> 1)) >> 4)
           | (((4 * v60 / 3 + 1) | ((unsigned __int64)(4 * v60 / 3 + 1) >> 1)) >> 2)
           | (4 * v60 / 3 + 1)
           | ((4 * v60 / 3 + 1) >> 1))
          + 1);
          v60 = v176;
          v62 = v174;
          v63 = &v188[(unsigned int)v189];
          v165 = (__int64)v188;
          if ( v188 == v63 )
            goto LABEL_135;
LABEL_113:
          while ( 2 )
          {
            v84 = *(v63 - 1);
            v85 = v180;
            v172 = v84;
            if ( (_DWORD)v180 )
            {
              v86 = 1;
              v85 = 1;
              v87 = (v180 - 1) & (((unsigned int)v84 >> 9) ^ ((unsigned int)v84 >> 4));
              v88 = v178[v87];
              if ( v84 != v88 )
              {
                while ( 1 )
                {
                  if ( v88 == -4096 )
                  {
                    v85 = 0;
                    goto LABEL_115;
                  }
                  v87 = (v180 - 1) & (v86 + v87);
                  v88 = v178[v87];
                  if ( v84 == v88 )
                    break;
                  ++v86;
                }
                v85 = 1;
              }
            }
LABEL_115:
            if ( !v60 )
            {
              ++v173;
              goto LABEL_117;
            }
            v64 = 1;
            v65 = 0;
            LODWORD(v66) = (v60 - 1) & (((unsigned int)v84 >> 9) ^ ((unsigned int)v84 >> 4));
            v67 = (__int64 *)(v62 + ((unsigned __int64)(unsigned int)v66 << 6));
            v68 = *v67;
            if ( v84 == *v67 )
            {
LABEL_98:
              v170 = v67 + 1;
              if ( v171 == v84 )
                goto LABEL_122;
            }
            else
            {
              while ( v68 != -4096 )
              {
                if ( !v65 && v68 == -8192 )
                  v65 = v67;
                v66 = (v60 - 1) & ((_DWORD)v66 + v64);
                v67 = (__int64 *)(v62 + (v66 << 6));
                v68 = *v67;
                if ( v84 == *v67 )
                  goto LABEL_98;
                ++v64;
              }
              if ( !v65 )
                v65 = v67;
              ++v173;
              v89 = v175 + 1;
              if ( 4 * ((int)v175 + 1) >= 3 * v60 )
              {
LABEL_117:
                sub_2730F10((__int64)&v173, 2 * v60);
                if ( !v176 )
                  goto LABEL_311;
                v84 = v172;
                v89 = v175 + 1;
                LODWORD(v90) = (v176 - 1) & (((unsigned int)v172 >> 9) ^ ((unsigned int)v172 >> 4));
                v65 = (__int64 *)(v174 + ((unsigned __int64)(unsigned int)v90 << 6));
                v91 = *v65;
                if ( *v65 != v172 )
                {
                  v160 = 1;
                  v161 = 0;
                  while ( v91 != -4096 )
                  {
                    if ( v91 == -8192 && !v161 )
                      v161 = v65;
                    v90 = (v176 - 1) & ((_DWORD)v90 + v160);
                    v65 = (__int64 *)(v174 + (v90 << 6));
                    v91 = *v65;
                    if ( v172 == *v65 )
                      goto LABEL_119;
                    ++v160;
                  }
                  if ( v161 )
                    v65 = v161;
                }
              }
              else if ( v60 - (v89 + HIDWORD(v175)) <= v60 >> 3 )
              {
                sub_2730F10((__int64)&v173, v60);
                if ( !v176 )
                {
LABEL_311:
                  LODWORD(v175) = v175 + 1;
                  BUG();
                }
                v138 = 1;
                v89 = v175 + 1;
                v139 = 0;
                LODWORD(v140) = (v176 - 1) & (((unsigned int)v172 >> 9) ^ ((unsigned int)v172 >> 4));
                v65 = (__int64 *)(v174 + ((unsigned __int64)(unsigned int)v140 << 6));
                v84 = *v65;
                if ( v172 != *v65 )
                {
                  while ( v84 != -4096 )
                  {
                    if ( !v139 && v84 == -8192 )
                      v139 = v65;
                    v140 = (v176 - 1) & ((_DWORD)v140 + v138);
                    v65 = (__int64 *)(v174 + (v140 << 6));
                    v84 = *v65;
                    if ( v172 == *v65 )
                      goto LABEL_119;
                    ++v138;
                  }
                  v84 = v172;
                  if ( v139 )
                    v65 = v139;
                }
              }
LABEL_119:
              LODWORD(v175) = v89;
              if ( *v65 != -4096 )
                --HIDWORD(v175);
              *v65 = v84;
              v170 = v65 + 1;
              v84 = v172;
              v65[5] = (__int64)(v65 + 7);
              v65[6] = 0;
              v65[7] = 0;
              *(_OWORD *)(v65 + 1) = 0;
              *(_OWORD *)(v65 + 3) = 0;
              if ( v171 == v84 )
              {
LABEL_122:
                ++v177;
                if ( !(_DWORD)v179 )
                {
                  if ( HIDWORD(v179) )
                  {
                    v92 = (unsigned int)v180;
                    if ( (unsigned int)v180 <= 0x40 )
                      goto LABEL_125;
                    sub_C7D6A0((__int64)v178, 8LL * (unsigned int)v180, 8);
                    v84 = v172;
                    v178 = 0;
                    v179 = 0;
                    LODWORD(v180) = 0;
                  }
LABEL_129:
                  LODWORD(v182) = 0;
                  if ( (unsigned __int64)sub_FDD860(v166, v84) < v170[6] )
                    goto LABEL_257;
                  if ( sub_FDD860(v166, v172) != v170[6] )
                  {
                    v95 = *((unsigned int *)v170 + 10);
                    goto LABEL_132;
                  }
                  v95 = *((unsigned int *)v170 + 10);
                  if ( (unsigned int)v95 <= 1 )
                  {
LABEL_132:
                    v96 = (__int64 *)v170[4];
                    v97 = &v96[v95];
                    while ( v97 != v96 )
                    {
                      v98 = v96++;
                      sub_2733720((__int64)&v177, v98);
                    }
                    v60 = v176;
                    v62 = v174;
                  }
                  else
                  {
LABEL_257:
                    sub_2733720((__int64)&v177, &v171);
                    v60 = v176;
                    v62 = v174;
                  }
LABEL_135:
                  if ( v60 )
                  {
                    v99 = v62 + ((unsigned __int64)v60 << 6);
                    do
                    {
                      if ( *(_QWORD *)v62 != -8192 && *(_QWORD *)v62 != -4096 )
                      {
                        v100 = *(_QWORD *)(v62 + 40);
                        if ( v100 != v62 + 56 )
                          _libc_free(v100);
                        sub_C7D6A0(*(_QWORD *)(v62 + 16), 8LL * *(unsigned int *)(v62 + 32), 8);
                      }
                      v62 += 64;
                    }
                    while ( v99 != v62 );
                    v60 = v176;
                    v62 = v174;
                  }
LABEL_144:
                  sub_C7D6A0(v62, (unsigned __int64)v60 << 6, 8);
                  if ( v188 != v190 )
                    _libc_free((unsigned __int64)v188);
                  if ( !v194 )
                    _libc_free(v191.m128i_u64[1]);
                  if ( !v186 )
                    _libc_free((unsigned __int64)s);
                  v16 = v181;
                  v101 = &v181[(unsigned int)v182];
                  if ( v101 != v181 )
                  {
                    v102 = v181;
                    do
                    {
                      v105 = sub_AA5190(*v102);
                      if ( v105 )
                      {
                        v103 = v106;
                        v104 = HIBYTE(v106);
                      }
                      else
                      {
                        v104 = 0;
                        v103 = 0;
                      }
                      ++v102;
                      v191.m128i_i8[8] = v103;
                      v191.m128i_i64[0] = v105;
                      v191.m128i_i8[9] = v104;
                      sub_2733BC0(a1, &v191);
                    }
                    while ( v101 != v102 );
                    goto LABEL_166;
                  }
                  goto LABEL_9;
                }
                v148 = 4 * v179;
                v92 = (unsigned int)v180;
                if ( (unsigned int)(4 * v179) < 0x40 )
                  v148 = 64;
                if ( v148 >= (unsigned int)v180 )
                {
LABEL_125:
                  v93 = v178;
                  v94 = &v178[v92];
                  if ( v178 != v94 )
                  {
                    do
                      *v93++ = -4096;
                    while ( v94 != v93 );
                    v84 = v172;
                  }
                  v179 = 0;
                  goto LABEL_129;
                }
                v149 = v178;
                v150 = (unsigned int)v180;
                if ( (_DWORD)v179 != 1 )
                {
                  _BitScanReverse(&v151, v179 - 1);
                  v152 = 1 << (33 - (v151 ^ 0x1F));
                  if ( v152 < 64 )
                    v152 = 64;
                  if ( v152 == (_DWORD)v180 )
                  {
                    v179 = 0;
                    v162 = &v178[v150];
                    do
                    {
                      if ( v149 )
                        *v149 = -4096;
                      ++v149;
                    }
                    while ( v162 != v149 );
                  }
                  else
                  {
                    v153 = (4 * v152 / 3u + 1) | ((unsigned __int64)(4 * v152 / 3u + 1) >> 1);
                    v154 = ((v153 | (v153 >> 2)) >> 4)
                         | v153
                         | (v153 >> 2)
                         | ((((v153 | (v153 >> 2)) >> 4) | v153 | (v153 >> 2)) >> 8);
                    v155 = (v154 | (v154 >> 16)) + 1;
                    v156 = 8 * ((v154 | (v154 >> 16)) + 1);
LABEL_268:
                    sub_C7D6A0((__int64)v178, v150 * 8, 8);
                    LODWORD(v180) = v155;
                    v157 = (_QWORD *)sub_C7D670(v156, 8);
                    v179 = 0;
                    v178 = v157;
                    for ( i = &v157[(unsigned int)v180]; i != v157; ++v157 )
                    {
                      if ( v157 )
                        *v157 = -4096;
                    }
                  }
                  v84 = v172;
                  goto LABEL_129;
                }
                v156 = 1024;
                v155 = 128;
                goto LABEL_268;
              }
            }
            if ( v84 )
            {
              v69 = (unsigned int)(*(_DWORD *)(v84 + 44) + 1);
              v70 = *(_DWORD *)(v84 + 44) + 1;
            }
            else
            {
              v69 = 0;
              v70 = 0;
            }
            if ( v70 >= *(_DWORD *)(v21 + 32) )
              BUG();
            v71 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v21 + 24) + 8 * v69) + 8LL);
            if ( v176 )
            {
              v72 = 1;
              v73 = 0;
              v74 = (v176 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
              v75 = (_QWORD *)(v174 + ((unsigned __int64)v74 << 6));
              v76 = *v75;
              if ( v71 == *v75 )
              {
LABEL_104:
                v77 = (__int64)(v75 + 1);
                goto LABEL_105;
              }
              while ( v76 != -4096 )
              {
                if ( v76 == -8192 && !v73 )
                  v73 = v75;
                v74 = (v176 - 1) & (v72 + v74);
                v75 = (_QWORD *)(v174 + ((unsigned __int64)v74 << 6));
                v76 = *v75;
                if ( v71 == *v75 )
                  goto LABEL_104;
                ++v72;
              }
              if ( !v73 )
                v73 = v75;
              ++v173;
              v137 = v175 + 1;
              if ( 4 * ((int)v175 + 1) < 3 * v176 )
              {
                if ( v176 - HIDWORD(v175) - v137 <= v176 >> 3 )
                {
                  sub_2730F10((__int64)&v173, v176);
                  if ( !v176 )
                  {
LABEL_310:
                    LODWORD(v175) = v175 + 1;
                    BUG();
                  }
                  v145 = 1;
                  v144 = 0;
                  LODWORD(v146) = (v176 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
                  v137 = v175 + 1;
                  v73 = (_QWORD *)(v174 + ((unsigned __int64)(unsigned int)v146 << 6));
                  v147 = *v73;
                  if ( v71 != *v73 )
                  {
                    while ( v147 != -4096 )
                    {
                      if ( !v144 && v147 == -8192 )
                        v144 = v73;
                      v146 = (v176 - 1) & ((_DWORD)v146 + v145);
                      v73 = (_QWORD *)(v174 + (v146 << 6));
                      v147 = *v73;
                      if ( v71 == *v73 )
                        goto LABEL_217;
                      ++v145;
                    }
                    goto LABEL_251;
                  }
                }
                goto LABEL_217;
              }
            }
            else
            {
              ++v173;
            }
            sub_2730F10((__int64)&v173, 2 * v176);
            if ( !v176 )
              goto LABEL_310;
            LODWORD(v141) = (v176 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
            v137 = v175 + 1;
            v73 = (_QWORD *)(v174 + ((unsigned __int64)(unsigned int)v141 << 6));
            v142 = *v73;
            if ( v71 != *v73 )
            {
              v143 = 1;
              v144 = 0;
              while ( v142 != -4096 )
              {
                if ( !v144 && v142 == -8192 )
                  v144 = v73;
                v141 = (v176 - 1) & ((_DWORD)v141 + v143);
                v73 = (_QWORD *)(v174 + (v141 << 6));
                v142 = *v73;
                if ( v71 == *v73 )
                  goto LABEL_217;
                ++v143;
              }
LABEL_251:
              if ( v144 )
                v73 = v144;
            }
LABEL_217:
            LODWORD(v175) = v137;
            if ( *v73 != -4096 )
              --HIDWORD(v175);
            *v73 = v71;
            v77 = (__int64)(v73 + 1);
            v73[5] = v73 + 7;
            v73[6] = 0;
            v73[7] = 0;
            *(_OWORD *)(v73 + 1) = 0;
            *(_OWORD *)(v73 + 3) = 0;
LABEL_105:
            if ( v85 )
              goto LABEL_110;
            v78 = sub_AA4FF0(v172);
            if ( !v78 )
              BUG();
            v79 = (unsigned int)*(unsigned __int8 *)(v78 - 24) - 39;
            if ( (unsigned int)v79 <= 0x38 )
            {
              v80 = 0x100060000000001LL;
              if ( _bittest64(&v80, v79) )
                goto LABEL_157;
            }
            if ( (unsigned __int64)sub_FDD860(v166, v172) < v170[6] )
              goto LABEL_110;
            if ( sub_FDD860(v166, v172) != v170[6] )
            {
LABEL_157:
              v107 = *((unsigned int *)v170 + 10);
              goto LABEL_158;
            }
            v107 = *((unsigned int *)v170 + 10);
            if ( (unsigned int)v107 > 1 )
            {
LABEL_110:
              sub_2733720(v77, &v172);
              v81 = sub_FDD860(v166, v172);
              v82 = __CFADD__(*(_QWORD *)(v77 + 48), v81);
              v83 = *(_QWORD *)(v77 + 48) + v81;
              if ( v82 )
                goto LABEL_161;
            }
            else
            {
LABEL_158:
              v108 = (__int64 *)v170[4];
              v109 = &v108[v107];
              while ( v109 != v108 )
              {
                v110 = v108++;
                sub_2733720(v77, v110);
              }
              v111 = *(_QWORD *)(v77 + 48);
              v82 = __CFADD__(v170[6], v111);
              v83 = v170[6] + v111;
              if ( v82 )
              {
LABEL_161:
                *(_QWORD *)(v77 + 48) = -1;
LABEL_112:
                v60 = v176;
                v62 = v174;
                if ( --v63 == (_QWORD *)v165 )
                  goto LABEL_135;
                continue;
              }
            }
            break;
          }
          *(_QWORD *)(v77 + 48) = v83;
          goto LABEL_112;
        }
      }
    }
    v19 = &v183;
  }
  else
  {
    v9 = a4;
    do
    {
      if ( !*(_QWORD *)v9 )
        BUG();
      v10 = *(_QWORD *)(*(_QWORD *)v9 + 16LL);
      v9 += 16;
      v191.m128i_i64[0] = v10;
      sub_2733720((__int64)&v177, v191.m128i_i64);
    }
    while ( v8 != v9 );
    v11 = v180;
    v7 = (__int64)a2;
    v6 = (__int64)v178;
    v12 = a2[5];
    if ( (_DWORD)v180 )
    {
      v13 = (v180 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v14 = v178[v13];
      if ( v12 == v14 )
      {
LABEL_7:
        v15 = *(_QWORD *)(v12 + 56);
        goto LABEL_8;
      }
      a6 = 1;
      while ( v14 != -4096 )
      {
        v13 = (v180 - 1) & (a6 + v13);
        v14 = v178[v13];
        if ( v12 == v14 )
          goto LABEL_7;
        a6 = (unsigned int)(a6 + 1);
      }
    }
    v18 = v182;
    v19 = v181;
    v166 = (__int64 *)a2[2];
    v20 = (unsigned int)v182;
    if ( v166 )
      goto LABEL_15;
    if ( (unsigned int)v182 > 1 )
    {
      v114 = (__int64)v178;
      v115 = v181[(unsigned int)v182 - 1];
      if ( !(_DWORD)v180 )
        goto LABEL_195;
LABEL_172:
      v116 = v11 - 1;
      v117 = v116 & (((unsigned int)v115 >> 9) ^ ((unsigned int)v115 >> 4));
      v118 = (__int64 *)(v6 + 8LL * v117);
      v119 = *v118;
      if ( *v118 == v115 )
      {
LABEL_173:
        *v118 = -8192;
        LODWORD(v179) = v179 - 1;
        v114 = (__int64)v178;
        ++HIDWORD(v179);
        LODWORD(v182) = v182 - 1;
        LODWORD(v120) = v182;
        v121 = v181[(unsigned int)v182 - 1];
        if ( !(_DWORD)v180 )
          goto LABEL_177;
        v116 = v180 - 1;
      }
      else
      {
        v134 = 1;
        while ( v119 != -4096 )
        {
          v159 = v134 + 1;
          v117 = v116 & (v117 + v134);
          v118 = (__int64 *)(v6 + 8LL * v117);
          v119 = *v118;
          if ( v115 == *v118 )
            goto LABEL_173;
          v134 = v159;
        }
        LODWORD(v182) = v18 - 1;
        v120 = (unsigned int)(v18 - 1);
        v121 = v19[v120 - 1];
      }
      v122 = v116 & (((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4));
      v123 = (__int64 *)(v114 + 8LL * v122);
      v124 = *v123;
      if ( v121 == *v123 )
      {
LABEL_176:
        *v123 = -8192;
        LODWORD(v120) = v182;
        LODWORD(v179) = v179 - 1;
        ++HIDWORD(v179);
        goto LABEL_177;
      }
      v135 = 1;
      while ( v124 != -4096 )
      {
        v136 = v135 + 1;
        v122 = v116 & (v122 + v135);
        v123 = (__int64 *)(v114 + 8LL * v122);
        v124 = *v123;
        if ( v121 == *v123 )
          goto LABEL_176;
        v135 = v136;
      }
      while ( 1 )
      {
LABEL_177:
        LODWORD(v182) = v120 - 1;
        v125 = *(_QWORD *)(*(_QWORD *)(v115 + 72) + 80LL);
        if ( v125 )
          v125 -= 24;
        if ( v125 != v115 && v125 != v121 )
        {
          v126 = a2[1];
          v127 = 0;
          v128 = *(_DWORD *)(v126 + 32);
          v129 = (unsigned int)(*(_DWORD *)(v115 + 44) + 1);
          if ( (unsigned int)v129 < v128 )
            v127 = *(_QWORD *)(*(_QWORD *)(v126 + 24) + 8 * v129);
          if ( v121 )
          {
            v130 = (unsigned int)(*(_DWORD *)(v121 + 44) + 1);
            v131 = v130;
          }
          else
          {
            v130 = 0;
            v131 = 0;
          }
          v132 = 0;
          if ( v128 > v131 )
            v132 = *(_QWORD *)(*(_QWORD *)(v126 + 24) + 8 * v130);
          for ( ; v127 != v132; v127 = *(_QWORD *)(v127 + 8) )
          {
            if ( *(_DWORD *)(v127 + 16) < *(_DWORD *)(v132 + 16) )
            {
              v133 = v127;
              v127 = v132;
              v132 = v133;
            }
          }
          v125 = *(_QWORD *)v132;
        }
        v188 = (_QWORD *)v125;
        if ( a2[5] == v125 )
          break;
        sub_2733720((__int64)&v177, (__int64 *)&v188);
        v18 = v182;
        v19 = v181;
        if ( (unsigned int)v182 <= 1 )
          goto LABEL_163;
        v6 = (__int64)v178;
        v11 = v180;
        v115 = v181[(unsigned int)v182 - 1];
        v114 = (__int64)v178;
        if ( (_DWORD)v180 )
          goto LABEL_172;
LABEL_195:
        LODWORD(v182) = v18 - 1;
        v120 = (unsigned int)(v18 - 1);
        v121 = v19[v120 - 1];
      }
      v15 = *(_QWORD *)(v125 + 56);
LABEL_8:
      v191.m128i_i64[0] = v15;
      v191.m128i_i16[4] = 1;
      sub_2733BC0(a1, &v191);
      v16 = v181;
      goto LABEL_9;
    }
  }
LABEL_163:
  v112 = *(unsigned __int8 **)(*v19 + 56);
  if ( v112 )
    v112 -= 24;
  v191.m128i_i64[0] = (__int64)sub_27306B0(v7, v112, 0xFFFFFFFF);
  v191.m128i_i64[1] = v113;
  sub_2733BC0(a1, &v191);
LABEL_166:
  v16 = v181;
LABEL_9:
  if ( v16 != &v183 )
    _libc_free((unsigned __int64)v16);
  sub_C7D6A0((__int64)v178, 8LL * (unsigned int)v180, 8);
  return a1;
}

// Function: sub_30A9120
// Address: 0x30a9120
//
_QWORD *__fastcall sub_30A9120(_QWORD *a1, __int64 a2, __int64 *a3)
{
  _QWORD *v3; // r15
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rsi
  char v10; // dl
  char v11; // al
  __int64 v12; // rax
  __int64 v13; // r13
  __m128i *v14; // rax
  __int64 v15; // r8
  _BYTE *v16; // r13
  _BYTE *v17; // r15
  unsigned __int64 v18; // rdi
  __int64 v19; // r13
  unsigned __int64 v20; // rbx
  volatile signed __int32 *v21; // r12
  signed __int32 v22; // eax
  signed __int32 v23; // eax
  __int64 v24; // rbx
  unsigned __int64 v25; // r12
  volatile signed __int32 *v26; // r13
  signed __int32 v27; // eax
  signed __int32 v28; // eax
  __int64 v29; // r14
  __m128i *v30; // rax
  __int64 *v31; // rbx
  __int64 v32; // rdi
  unsigned __int64 v33; // rax
  __int64 *v34; // rdx
  __int64 *v35; // rdi
  __int64 v36; // rcx
  __int64 v37; // rdi
  _QWORD *v38; // rdx
  __int64 v39; // r8
  _QWORD *v40; // r11
  int v41; // r15d
  int v42; // edi
  bool v43; // bl
  __int64 *v44; // r12
  __int64 v45; // rdi
  unsigned int v46; // edx
  __int64 v47; // r8
  unsigned __int64 v48; // rdx
  _QWORD *v49; // rcx
  _QWORD *v50; // rax
  bool v51; // al
  __int64 *v52; // rcx
  __int64 *v53; // r9
  _QWORD *v54; // rdi
  int *v55; // r12
  int *v56; // rax
  unsigned __int64 v57; // rbx
  int *v58; // r13
  unsigned __int64 v59; // r15
  unsigned __int64 v60; // r12
  unsigned __int64 v61; // r14
  unsigned __int64 v62; // rdi
  _QWORD *v63; // rax
  __int64 v64; // rax
  unsigned int v65; // r10d
  __int64 v66; // rbx
  __int64 v67; // rax
  __int64 v68; // r12
  __int64 v69; // rax
  __int64 v70; // rcx
  __int64 v71; // r14
  __int64 v72; // rax
  __int64 v73; // rcx
  __int64 v74; // rdx
  __int64 v75; // rax
  bool v76; // cc
  _QWORD *v77; // rax
  __int64 v78; // r13
  __int64 v79; // rax
  __int64 v80; // rbx
  __int64 v81; // rsi
  __int64 v82; // rax
  __int64 v83; // rcx
  __int64 v84; // rax
  char *v85; // rax
  __int64 v86; // rdx
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 *v89; // rcx
  __int64 v90; // rdx
  _QWORD *v91; // rax
  unsigned __int64 v92; // r12
  _QWORD *v93; // r14
  __int64 v94; // r15
  int v95; // ebx
  unsigned __int64 v96; // r13
  _QWORD *v97; // rdi
  unsigned __int64 v98; // r12
  __int64 v99; // r15
  int v100; // ebx
  unsigned __int64 v101; // r13
  _QWORD *v102; // rdi
  _QWORD *v103; // rdx
  _QWORD *v104; // rax
  int v105; // ecx
  _QWORD *v106; // rdx
  _QWORD *v107; // rax
  int v108; // ecx
  __int64 v109; // rdx
  _QWORD *v110; // rax
  int v111; // ecx
  int v112; // r11d
  __int64 v113; // rcx
  __int64 v114; // r9
  int v115; // r8d
  int v116; // r11d
  __int64 v117; // rcx
  __int64 v118; // r9
  int v119; // r8d
  int v120; // edx
  _QWORD *v121; // [rsp+18h] [rbp-548h]
  _QWORD *v122; // [rsp+20h] [rbp-540h]
  __int64 *v123; // [rsp+28h] [rbp-538h]
  __int64 *v124; // [rsp+40h] [rbp-520h]
  char v125; // [rsp+48h] [rbp-518h]
  __int64 *v127; // [rsp+58h] [rbp-508h]
  __int64 *v128; // [rsp+58h] [rbp-508h]
  int v129; // [rsp+58h] [rbp-508h]
  unsigned __int64 v130; // [rsp+58h] [rbp-508h]
  _QWORD *v131; // [rsp+60h] [rbp-500h]
  unsigned __int64 v132; // [rsp+60h] [rbp-500h]
  unsigned __int64 v133; // [rsp+60h] [rbp-500h]
  _BYTE *v134; // [rsp+68h] [rbp-4F8h]
  __int64 v136; // [rsp+68h] [rbp-4F8h]
  _QWORD v137[2]; // [rsp+70h] [rbp-4F0h] BYREF
  char v138; // [rsp+80h] [rbp-4E0h]
  __int64 v139; // [rsp+90h] [rbp-4D0h] BYREF
  __int64 v140; // [rsp+98h] [rbp-4C8h]
  __int64 v141; // [rsp+A0h] [rbp-4C0h]
  __int64 v142; // [rsp+A8h] [rbp-4B8h]
  __int64 v143; // [rsp+B0h] [rbp-4B0h] BYREF
  __int64 v144; // [rsp+B8h] [rbp-4A8h] BYREF
  _QWORD *v145; // [rsp+C0h] [rbp-4A0h]
  __int64 *v146; // [rsp+C8h] [rbp-498h]
  __int64 *v147; // [rsp+D0h] [rbp-490h]
  __int64 v148; // [rsp+D8h] [rbp-488h]
  int v149; // [rsp+E8h] [rbp-478h] BYREF
  _QWORD *v150; // [rsp+F0h] [rbp-470h]
  int *v151; // [rsp+F8h] [rbp-468h]
  int *v152; // [rsp+100h] [rbp-460h]
  __int64 v153; // [rsp+108h] [rbp-458h]
  char v154; // [rsp+110h] [rbp-450h]
  _QWORD **v155; // [rsp+120h] [rbp-440h] BYREF
  int v156; // [rsp+128h] [rbp-438h] BYREF
  _QWORD *v157; // [rsp+130h] [rbp-430h] BYREF
  int *v158; // [rsp+138h] [rbp-428h]
  int *v159; // [rsp+140h] [rbp-420h]
  __int64 v160; // [rsp+148h] [rbp-418h]
  int v161; // [rsp+158h] [rbp-408h] BYREF
  _QWORD *v162; // [rsp+160h] [rbp-400h]
  int *v163; // [rsp+168h] [rbp-3F8h]
  int *v164; // [rsp+170h] [rbp-3F0h]
  __int64 v165; // [rsp+178h] [rbp-3E8h]
  __int64 v166; // [rsp+180h] [rbp-3E0h] BYREF
  int v167; // [rsp+188h] [rbp-3D8h] BYREF
  __int64 v168; // [rsp+190h] [rbp-3D0h]
  int *v169; // [rsp+198h] [rbp-3C8h]
  int *v170; // [rsp+1A0h] [rbp-3C0h]
  __int64 v171; // [rsp+1A8h] [rbp-3B8h]
  __m128i *v172; // [rsp+1B0h] [rbp-3B0h] BYREF
  __int8 *v173; // [rsp+1B8h] [rbp-3A8h] BYREF
  __m128i v174; // [rsp+1C0h] [rbp-3A0h] BYREF
  __int64 v175; // [rsp+1D8h] [rbp-388h]
  __int64 *v176; // [rsp+1E0h] [rbp-380h]
  __int64 v177; // [rsp+1E8h] [rbp-378h]
  char *v178; // [rsp+1F0h] [rbp-370h] BYREF
  __int64 v179; // [rsp+1F8h] [rbp-368h]
  _BYTE v180[136]; // [rsp+200h] [rbp-360h] BYREF
  int v181; // [rsp+288h] [rbp-2D8h] BYREF
  unsigned __int64 v182; // [rsp+290h] [rbp-2D0h]
  int *v183; // [rsp+298h] [rbp-2C8h]
  int *v184; // [rsp+2A0h] [rbp-2C0h]
  __int64 v185; // [rsp+2A8h] [rbp-2B8h]
  __m128i *v186; // [rsp+2B0h] [rbp-2B0h] BYREF
  unsigned __int64 v187; // [rsp+2B8h] [rbp-2A8h]
  __m128i v188; // [rsp+2C0h] [rbp-2A0h] BYREF
  _QWORD v189[2]; // [rsp+2D0h] [rbp-290h] BYREF
  __int64 v190; // [rsp+2E0h] [rbp-280h] BYREF
  __int64 *v191; // [rsp+2E8h] [rbp-278h]
  __int64 v192; // [rsp+2F0h] [rbp-270h]
  unsigned __int64 v193[2]; // [rsp+2F8h] [rbp-268h] BYREF
  _BYTE v194[136]; // [rsp+308h] [rbp-258h] BYREF
  int v195; // [rsp+390h] [rbp-1D0h] BYREF
  unsigned __int64 v196; // [rsp+398h] [rbp-1C8h]
  int *v197; // [rsp+3A0h] [rbp-1C0h]
  int *v198; // [rsp+3A8h] [rbp-1B8h]
  __int64 v199; // [rsp+3B0h] [rbp-1B0h]
  __m128i **v200; // [rsp+3C0h] [rbp-1A0h] BYREF
  __int64 v201; // [rsp+3C8h] [rbp-198h]
  __int64 v202; // [rsp+3D0h] [rbp-190h]
  __int64 v203; // [rsp+3D8h] [rbp-188h]
  __int64 v204; // [rsp+3E0h] [rbp-180h]
  __int64 v205; // [rsp+3E8h] [rbp-178h]
  __int64 v206; // [rsp+3F0h] [rbp-170h]
  unsigned __int64 v207; // [rsp+3F8h] [rbp-168h]
  __int64 v208; // [rsp+400h] [rbp-160h]
  __int64 v209; // [rsp+408h] [rbp-158h]
  _BYTE *v210; // [rsp+410h] [rbp-150h]
  __int64 v211; // [rsp+418h] [rbp-148h]
  _BYTE v212[256]; // [rsp+420h] [rbp-140h] BYREF
  __int64 v213; // [rsp+520h] [rbp-40h]

  v3 = a1;
  v125 = *(_BYTE *)(a2 + 16);
  if ( !v125 )
  {
    memset(a1, 0, 0x90u);
    a1[3] = a1 + 1;
    a1[4] = a1 + 1;
    a1[9] = a1 + 7;
    a1[10] = a1 + 7;
    a1[15] = a1 + 13;
    a1[16] = a1 + 13;
    return v3;
  }
  v5 = *(_QWORD *)a2;
  LOWORD(v204) = 261;
  v200 = (__m128i **)v5;
  v201 = *(_QWORD *)(a2 + 8);
  sub_C7EA90((__int64)v137, (__int64 *)&v200, 0, 1u, 0, 0);
  if ( (v138 & 1) == 0 || !LODWORD(v137[0]) )
  {
    v6 = *(_QWORD *)(v137[0] + 16LL);
    v7 = *(_QWORD *)(v137[0] + 8LL);
    if ( (unsigned __int64)(v6 - v7) <= 3 )
    {
      v201 = *(_QWORD *)(v137[0] + 16LL) - v7;
      v8 = 0;
      v200 = (__m128i **)v7;
    }
    else
    {
      v200 = *(__m128i ***)(v137[0] + 8LL);
      v8 = v6 - v7 - 4;
      v6 = v7 + 4;
      v201 = 4;
    }
    v203 = v8;
    v9 = (__int64)&v200;
    v206 = 0x200000000LL;
    v210 = v212;
    v211 = 0x800000000LL;
    v202 = v6;
    v204 = 0;
    v205 = 0;
    v207 = 0;
    v208 = 0;
    v209 = 0;
    v213 = 0;
    sub_31568A0(&v143, &v200);
    v10 = v154 & 1;
    v11 = (2 * (v154 & 1)) | v154 & 0xFD;
    v154 = v11;
    if ( v10 )
    {
      v154 = v11 & 0xFD;
      v12 = v143;
      v13 = *a3;
      v143 = 0;
      v139 = v12 | 1;
      sub_C64870((__int64)&v155, &v139);
      v14 = (__m128i *)sub_2241130((unsigned __int64 *)&v155, 0, 0, "contextual profile file is invalid: ", 0x24u);
      v172 = &v174;
      if ( (__m128i *)v14->m128i_i64[0] == &v14[1] )
      {
        v174 = _mm_loadu_si128(v14 + 1);
      }
      else
      {
        v172 = (__m128i *)v14->m128i_i64[0];
        v174.m128i_i64[0] = v14[1].m128i_i64[0];
      }
      v9 = (__int64)&v186;
      v173 = (__int8 *)v14->m128i_i64[1];
      v14->m128i_i64[0] = (__int64)v14[1].m128i_i64;
      v14->m128i_i64[1] = 0;
      v14[1].m128i_i8[0] = 0;
      LOWORD(v189[0]) = 260;
      v186 = (__m128i *)&v172;
      sub_B6ECE0(v13, (__int64)&v186);
      if ( v172 != &v174 )
      {
        v9 = v174.m128i_i64[0] + 1;
        j_j___libc_free_0((unsigned __int64)v172);
      }
      if ( v155 != &v157 )
      {
        v9 = (__int64)v157 + 1;
        j_j___libc_free_0((unsigned __int64)v155);
      }
      if ( (v139 & 1) != 0 || (v139 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v139, v9);
      memset(a1, 0, 0x90u);
      a1[3] = a1 + 1;
      a1[4] = a1 + 1;
      a1[9] = a1 + 7;
      a1[10] = a1 + 7;
      a1[15] = a1 + 13;
      a1[16] = a1 + 13;
LABEL_17:
      if ( (v154 & 2) != 0 )
        goto LABEL_96;
      if ( (v154 & 1) != 0 )
      {
        if ( v143 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v143 + 8LL))(v143);
      }
      else
      {
        sub_30A7180(v150);
        sub_30A7670(v145);
      }
      v15 = 32LL * (unsigned int)v211;
      v134 = v210;
      v16 = &v210[v15];
      if ( v210 != &v210[v15] )
      {
        v131 = v3;
        v17 = &v210[v15];
        do
        {
          v18 = *((_QWORD *)v17 - 3);
          v19 = *((_QWORD *)v17 - 2);
          v17 -= 32;
          v20 = v18;
          if ( v19 != v18 )
          {
            do
            {
              while ( 1 )
              {
                v21 = *(volatile signed __int32 **)(v20 + 8);
                if ( v21 )
                {
                  if ( &_pthread_key_create )
                  {
                    v22 = _InterlockedExchangeAdd(v21 + 2, 0xFFFFFFFF);
                  }
                  else
                  {
                    v22 = *((_DWORD *)v21 + 2);
                    *((_DWORD *)v21 + 2) = v22 - 1;
                  }
                  if ( v22 == 1 )
                  {
                    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v21 + 16LL))(v21);
                    if ( &_pthread_key_create )
                    {
                      v23 = _InterlockedExchangeAdd(v21 + 3, 0xFFFFFFFF);
                    }
                    else
                    {
                      v23 = *((_DWORD *)v21 + 3);
                      *((_DWORD *)v21 + 3) = v23 - 1;
                    }
                    if ( v23 == 1 )
                      break;
                  }
                }
                v20 += 16LL;
                if ( v19 == v20 )
                  goto LABEL_33;
              }
              v20 += 16LL;
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v21 + 24LL))(v21);
            }
            while ( v19 != v20 );
LABEL_33:
            v18 = *((_QWORD *)v17 + 1);
          }
          if ( v18 )
            j_j___libc_free_0(v18);
        }
        while ( v134 != v17 );
        v3 = v131;
        v16 = v210;
      }
      if ( v16 != v212 )
        _libc_free((unsigned __int64)v16);
      v24 = v208;
      v25 = v207;
      if ( v208 != v207 )
      {
        do
        {
          v26 = *(volatile signed __int32 **)(v25 + 8);
          if ( v26 )
          {
            if ( &_pthread_key_create )
            {
              v27 = _InterlockedExchangeAdd(v26 + 2, 0xFFFFFFFF);
            }
            else
            {
              v27 = *((_DWORD *)v26 + 2);
              *((_DWORD *)v26 + 2) = v27 - 1;
            }
            if ( v27 == 1 )
            {
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v26 + 16LL))(v26);
              if ( &_pthread_key_create )
              {
                v28 = _InterlockedExchangeAdd(v26 + 3, 0xFFFFFFFF);
              }
              else
              {
                v28 = *((_DWORD *)v26 + 3);
                *((_DWORD *)v26 + 3) = v28 - 1;
              }
              if ( v28 == 1 )
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v26 + 24LL))(v26);
            }
          }
          v25 += 16LL;
        }
        while ( v24 != v25 );
        v25 = v207;
      }
      if ( v25 )
        j_j___libc_free_0(v25);
      goto LABEL_59;
    }
    v139 = 0;
    v140 = 0;
    v31 = (__int64 *)a3[4];
    v141 = 0;
    v142 = 0;
    v123 = a3 + 3;
    if ( a3 + 3 == v31 )
    {
LABEL_99:
      v124 = v146;
      if ( v146 != &v144 )
      {
        v122 = v3;
        v43 = (v154 & 2) != 0;
        do
        {
          v44 = v124;
          v124 = (__int64 *)sub_220EEE0((__int64)v124);
          if ( (_DWORD)v142 )
          {
            v45 = v44[4];
            v46 = (v142 - 1) & (((0xBF58476D1CE4E5B9LL * v45) >> 31) ^ (484763065 * v45));
            v9 = v140 + 8LL * v46;
            v47 = *(_QWORD *)v9;
            if ( v45 == *(_QWORD *)v9 )
            {
LABEL_104:
              if ( v9 != v140 + 8LL * (unsigned int)v142 )
                continue;
            }
            else
            {
              v9 = 1;
              while ( v47 != -1 )
              {
                v65 = v9 + 1;
                v46 = (v142 - 1) & (v9 + v46);
                v9 = v140 + 8LL * v46;
                v47 = *(_QWORD *)v9;
                if ( v45 == *(_QWORD *)v9 )
                  goto LABEL_104;
                v9 = v65;
              }
            }
          }
          if ( v43 )
            goto LABEL_96;
          if ( v145 )
          {
            v48 = v44[4];
            v9 = (__int64)&v144;
            v49 = v145;
            while ( 1 )
            {
              if ( v49[4] < v48 )
              {
                v49 = (_QWORD *)v49[3];
              }
              else
              {
                v50 = (_QWORD *)v49[2];
                if ( v49[4] <= v48 )
                {
                  v136 = (__int64)v49;
                  v52 = (__int64 *)v49[3];
                  v128 = (__int64 *)v9;
                  v53 = (__int64 *)v9;
                  if ( v52 )
                  {
                    do
                    {
                      v9 = v52[3];
                      if ( v48 >= v52[4] )
                      {
                        v52 = (__int64 *)v52[3];
                      }
                      else
                      {
                        v53 = v52;
                        v52 = (__int64 *)v52[2];
                      }
                    }
                    while ( v52 );
                    v128 = v53;
                  }
                  v54 = (_QWORD *)v136;
                  if ( v50 )
                  {
                    do
                    {
                      v9 = v50[2];
                      if ( v48 > v50[4] )
                      {
                        v50 = (_QWORD *)v50[3];
                      }
                      else
                      {
                        v54 = v50;
                        v50 = (_QWORD *)v50[2];
                      }
                    }
                    while ( v50 );
                    v136 = (__int64)v54;
                  }
                  v43 = v128 == &v144 && v146 == (__int64 *)v136;
                  if ( !v43 )
                  {
                    if ( v128 != (__int64 *)v136 )
                    {
                      do
                      {
                        v55 = (int *)v136;
                        v136 = sub_220EF30(v136);
                        v56 = sub_220F330(v55, &v144);
                        v57 = *((_QWORD *)v56 + 28);
                        v58 = v56;
                        while ( v57 )
                        {
                          v59 = v57;
                          sub_30A7420(*(_QWORD **)(v57 + 24));
                          v60 = *(_QWORD *)(v57 + 56);
                          v57 = *(_QWORD *)(v57 + 16);
                          while ( v60 )
                          {
                            v61 = v60;
                            sub_30A7670(*(_QWORD **)(v60 + 24));
                            v60 = *(_QWORD *)(v60 + 16);
                            sub_30A7730((_QWORD *)(v61 + 40));
                            j_j___libc_free_0(v61);
                          }
                          j_j___libc_free_0(v59);
                        }
                        v62 = *((_QWORD *)v58 + 8);
                        if ( (int *)v62 != v58 + 20 )
                          _libc_free(v62);
                        v63 = (_QWORD *)*((_QWORD *)v58 + 6);
                        if ( v63 )
                          *v63 = *((_QWORD *)v58 + 5);
                        v64 = *((_QWORD *)v58 + 5);
                        if ( v64 )
                          *(_QWORD *)(v64 + 8) = *((_QWORD *)v58 + 6);
                        v9 = 256;
                        j_j___libc_free_0((unsigned __int64)v58);
                        --v148;
                      }
                      while ( v128 != (__int64 *)v136 );
                      v43 = (v154 & 2) != 0;
                    }
                    goto LABEL_101;
                  }
LABEL_115:
                  sub_30A7670(v145);
                  v145 = 0;
                  v148 = 0;
                  v146 = &v144;
                  v147 = &v144;
                  v43 = (v154 & 2) != 0;
                  goto LABEL_101;
                }
                v9 = (__int64)v49;
                v49 = (_QWORD *)v49[2];
              }
              if ( !v49 )
              {
                v127 = (__int64 *)v9;
                v51 = v9 == (_QWORD)&v144;
                goto LABEL_114;
              }
            }
          }
          v51 = v125;
          v127 = &v144;
LABEL_114:
          v43 = v51 && v146 == v127;
          if ( v43 )
            goto LABEL_115;
LABEL_101:
          ;
        }
        while ( v124 != &v144 );
        v3 = v122;
        if ( v43 )
          goto LABEL_96;
      }
      if ( v148 )
      {
        v156 = 0;
        v158 = &v156;
        v159 = &v156;
        v169 = &v167;
        v170 = &v167;
        v157 = 0;
        v160 = 0;
        v66 = a3[4];
        v161 = 0;
        v162 = 0;
        v163 = &v161;
        v164 = &v161;
        v165 = 0;
        v167 = 0;
        v168 = 0;
        v171 = 0;
        if ( v123 != (__int64 *)v66 )
        {
          v121 = v3;
          do
          {
            v67 = 0;
            if ( v66 )
              v67 = v66 - 56;
            v68 = v67;
            if ( !sub_B2FC80(v67) )
            {
              v69 = sub_30A7A60(v68);
              v70 = *(_QWORD *)(v68 + 80);
              v71 = v69;
              if ( !v70 )
LABEL_273:
                BUG();
              v72 = *(_QWORD *)(v70 + 32);
              v73 = v70 + 24;
              if ( v73 != v72 )
              {
                while ( 1 )
                {
                  if ( !v72 )
LABEL_272:
                    BUG();
                  if ( *(_BYTE *)(v72 - 24) == 85 )
                  {
                    v74 = *(_QWORD *)(v72 - 56);
                    if ( v74 )
                    {
                      if ( !*(_BYTE *)v74 )
                      {
                        v9 = *(_QWORD *)(v72 + 56);
                        if ( *(_QWORD *)(v74 + 24) == v9
                          && (*(_BYTE *)(v74 + 33) & 0x20) != 0
                          && (unsigned int)(*(_DWORD *)(v74 + 36) - 198) <= 1 )
                        {
                          break;
                        }
                      }
                    }
                  }
                  v72 = *(_QWORD *)(v72 + 8);
                  if ( v73 == v72 )
                    goto LABEL_158;
                }
                v75 = sub_B59B70(v72 - 24);
                v76 = *(_DWORD *)(v75 + 32) <= 0x40u;
                v77 = *(_QWORD **)(v75 + 24);
                if ( !v76 )
                  v77 = (_QWORD *)*v77;
                v129 = (int)v77;
                if ( (_DWORD)v77 )
                {
                  LODWORD(v78) = 0;
                  if ( v68 + 72 != *(_QWORD *)(v68 + 80) )
                  {
                    v79 = v66;
                    v80 = *(_QWORD *)(v68 + 80);
                    v81 = v79;
                    do
                    {
                      if ( !v80 )
                        goto LABEL_273;
                      v82 = *(_QWORD *)(v80 + 32);
                      if ( v80 + 24 != v82 )
                      {
                        while ( 1 )
                        {
                          if ( !v82 )
                            goto LABEL_272;
                          if ( *(_BYTE *)(v82 - 24) == 85 )
                          {
                            v83 = *(_QWORD *)(v82 - 56);
                            if ( v83 )
                            {
                              if ( !*(_BYTE *)v83
                                && *(_QWORD *)(v83 + 24) == *(_QWORD *)(v82 + 56)
                                && (*(_BYTE *)(v83 + 33) & 0x20) != 0
                                && *(_DWORD *)(v83 + 36) == 196 )
                              {
                                break;
                              }
                            }
                          }
                          v82 = *(_QWORD *)(v82 + 8);
                          if ( v80 + 24 == v82 )
                            goto LABEL_191;
                        }
                        v84 = sub_B59B70(v82 - 24);
                        if ( *(_DWORD *)(v84 + 32) <= 0x40u )
                          v78 = *(_QWORD *)(v84 + 24);
                        else
                          v78 = **(_QWORD **)(v84 + 24);
                      }
LABEL_191:
                      v80 = *(_QWORD *)(v80 + 8);
                    }
                    while ( v68 + 72 != v80 );
                    v66 = v81;
                  }
                  v85 = (char *)sub_BD5D20(v68);
                  v172 = 0;
                  v173 = &v174.m128i_i8[8];
                  sub_30A69B0((__int64 *)&v173, v85, (__int64)&v85[v86]);
                  v179 = 0x1000000000LL;
                  v187 = (unsigned __int64)v172;
                  v175 = 0;
                  v178 = v180;
                  v176 = 0;
                  v177 = 0;
                  v181 = 0;
                  v182 = 0;
                  v183 = &v181;
                  v184 = &v181;
                  v185 = 0;
                  v186 = (__m128i *)v71;
                  v188.m128i_i64[0] = (__int64)v189;
                  sub_30A6F00(v188.m128i_i64, v173, (__int64)&v173[v174.m128i_i64[0]]);
                  v89 = v176;
                  v90 = v175;
                  v191 = v176;
                  v190 = v175;
                  if ( v176 )
                  {
                    *v176 = (__int64)&v190;
                    v90 = v175;
                  }
                  if ( v90 )
                  {
                    v89 = &v190;
                    *(_QWORD *)(v90 + 8) = &v190;
                  }
                  v176 = 0;
                  v193[0] = (unsigned __int64)v194;
                  v193[1] = 0x1000000000LL;
                  v175 = 0;
                  v192 = v177;
                  if ( (_DWORD)v179 )
                    sub_30A6A60((__int64)v193, &v178, v177, (__int64)v89, v87, v88);
                  if ( v182 )
                  {
                    v196 = v182;
                    v195 = v181;
                    v197 = v183;
                    v198 = v184;
                    *(_QWORD *)(v182 + 8) = &v195;
                    v182 = 0;
                    v199 = v185;
                    v183 = &v181;
                    v184 = &v181;
                    v185 = 0;
                  }
                  else
                  {
                    v195 = 0;
                    v196 = 0;
                    v197 = &v195;
                    v198 = &v195;
                    v199 = 0;
                  }
                  v9 = (__int64)&v186;
                  v91 = sub_30A8680(&v166, (__int64 *)&v186);
                  v92 = v196;
                  v93 = v91;
                  if ( v196 )
                  {
                    v94 = v66;
                    v95 = v78;
                    do
                    {
                      v96 = v92;
                      sub_30A7420(*(_QWORD **)(v92 + 24));
                      v97 = *(_QWORD **)(v92 + 56);
                      v92 = *(_QWORD *)(v92 + 16);
                      sub_30A7670(v97);
                      v9 = 88;
                      j_j___libc_free_0(v96);
                    }
                    while ( v92 );
                    LODWORD(v78) = v95;
                    v66 = v94;
                  }
                  if ( (_BYTE *)v193[0] != v194 )
                    _libc_free(v193[0]);
                  if ( v191 )
                    *v191 = v190;
                  if ( v190 )
                    *(_QWORD *)(v190 + 8) = v191;
                  if ( (_QWORD *)v188.m128i_i64[0] != v189 )
                  {
                    v9 = v189[0] + 1LL;
                    j_j___libc_free_0(v188.m128i_u64[0]);
                  }
                  v98 = v182;
                  if ( v182 )
                  {
                    v99 = v66;
                    v100 = v78;
                    do
                    {
                      v101 = v98;
                      sub_30A7420(*(_QWORD **)(v98 + 24));
                      v102 = *(_QWORD **)(v98 + 56);
                      v98 = *(_QWORD *)(v98 + 16);
                      sub_30A7670(v102);
                      v9 = 88;
                      j_j___libc_free_0(v101);
                    }
                    while ( v98 );
                    LODWORD(v78) = v100;
                    v66 = v99;
                  }
                  if ( v178 != v180 )
                    _libc_free((unsigned __int64)v178);
                  if ( v176 )
                    *v176 = v175;
                  if ( v175 )
                    *(_QWORD *)(v175 + 8) = v176;
                  if ( v173 != (__int8 *)&v174.m128i_u64[1] )
                  {
                    v9 = v174.m128i_i64[1] + 1;
                    j_j___libc_free_0((unsigned __int64)v173);
                  }
                  *((_DWORD *)v93 + 11) = v78;
                  *((_DWORD *)v93 + 10) = v129;
                }
              }
            }
LABEL_158:
            v66 = *(_QWORD *)(v66 + 8);
          }
          while ( v123 != (__int64 *)v66 );
          v3 = v121;
          if ( (v154 & 2) != 0 )
            goto LABEL_96;
        }
        sub_30A7670(v157);
        v157 = 0;
        v160 = 0;
        v158 = &v156;
        v159 = &v156;
        if ( v145 )
        {
          v157 = v145;
          v156 = v144;
          v158 = (int *)v146;
          v159 = (int *)v147;
          v145[1] = &v156;
          v145 = 0;
          v160 = v148;
          v148 = 0;
          v146 = &v144;
          v147 = &v144;
        }
        sub_30A7180(v162);
        v162 = 0;
        v163 = &v161;
        v164 = &v161;
        v165 = 0;
        if ( v150 )
        {
          v162 = v150;
          v161 = v149;
          v163 = v151;
          v164 = v152;
          v150[1] = &v161;
          v150 = 0;
          v165 = v153;
          v151 = &v149;
          v152 = &v149;
          v153 = 0;
        }
        sub_30A8D60((__int64)&v155);
        v103 = v157;
        v104 = v3 + 1;
        if ( v157 )
        {
          v105 = v156;
          v3[2] = v157;
          *((_DWORD *)v3 + 2) = v105;
          v3[3] = v158;
          v3[4] = v159;
          v103[1] = v104;
          v157 = 0;
          v3[5] = v160;
          v160 = 0;
          v158 = &v156;
          v159 = &v156;
        }
        else
        {
          *((_DWORD *)v3 + 2) = 0;
          v3[2] = 0;
          v3[3] = v104;
          v3[4] = v104;
          v3[5] = 0;
        }
        v106 = v162;
        v107 = v3 + 7;
        if ( v162 )
        {
          v108 = v161;
          v3[8] = v162;
          *((_DWORD *)v3 + 14) = v108;
          v3[9] = v163;
          v3[10] = v164;
          v106[1] = v107;
          v162 = 0;
          v3[11] = v165;
          v163 = &v161;
          v164 = &v161;
          v165 = 0;
        }
        else
        {
          *((_DWORD *)v3 + 14) = 0;
          v3[8] = 0;
          v3[9] = v107;
          v3[10] = v107;
          v3[11] = 0;
        }
        v109 = v168;
        v110 = v3 + 13;
        if ( v168 )
        {
          v111 = v167;
          v3[14] = v168;
          *((_DWORD *)v3 + 26) = v111;
          v3[15] = v169;
          v3[16] = v170;
          *(_QWORD *)(v109 + 8) = v110;
          v168 = 0;
          v3[17] = v171;
          v171 = 0;
          v169 = &v167;
          v170 = &v167;
        }
        else
        {
          *((_DWORD *)v3 + 26) = 0;
          v3[14] = 0;
          v3[15] = v110;
          v3[16] = v110;
          v3[17] = 0;
        }
        sub_30A77C0(0);
        sub_30A7180(v162);
        sub_30A7670(v157);
      }
      else
      {
        memset(v3, 0, 0x90u);
        v3[3] = v3 + 1;
        v3[4] = v3 + 1;
        v3[9] = v3 + 7;
        v3[10] = v3 + 7;
        v3[15] = v3 + 13;
        v3[16] = v3 + 13;
      }
      v9 = 8LL * (unsigned int)v142;
      sub_C7D6A0(v140, v9, 8);
      goto LABEL_17;
    }
    while ( 1 )
    {
      v32 = (__int64)(v31 - 7);
      if ( !v31 )
        v32 = 0;
      if ( sub_B2FC80(v32) )
        goto LABEL_65;
      v33 = sub_30A7A60(v32);
      if ( (v154 & 2) != 0 )
        goto LABEL_96;
      v34 = v145;
      if ( !v145 )
        goto LABEL_65;
      v35 = &v144;
      do
      {
        while ( 1 )
        {
          v9 = v34[2];
          v36 = v34[3];
          if ( v33 <= v34[4] )
            break;
          v34 = (__int64 *)v34[3];
          if ( !v36 )
            goto LABEL_75;
        }
        v35 = v34;
        v34 = (__int64 *)v34[2];
      }
      while ( v9 );
LABEL_75:
      if ( v35 == &v144 || v33 < v35[4] )
        goto LABEL_65;
      v9 = (unsigned int)v142;
      if ( !(_DWORD)v142 )
        break;
      v37 = ((unsigned int)((0xBF58476D1CE4E5B9LL * v33) >> 31) ^ (484763065 * (_DWORD)v33)) & ((_DWORD)v142 - 1);
      v38 = (_QWORD *)(v140 + 8 * v37);
      v39 = *v38;
      if ( v33 != *v38 )
      {
        v40 = 0;
        v41 = 1;
        while ( v39 != -1 )
        {
          if ( v39 == -2 && !v40 )
            v40 = v38;
          v120 = v41++;
          LODWORD(v37) = (v142 - 1) & (v120 + v37);
          v38 = (_QWORD *)(v140 + 8LL * (unsigned int)v37);
          v39 = *v38;
          if ( v33 == *v38 )
            goto LABEL_65;
        }
        if ( v40 )
          v38 = v40;
        ++v139;
        v42 = v141 + 1;
        if ( 4 * ((int)v141 + 1) < (unsigned int)(3 * v142) )
        {
          if ( (int)v142 - HIDWORD(v141) - v42 <= (unsigned int)v142 >> 3 )
          {
            v130 = ((0xBF58476D1CE4E5B9LL * v33) >> 31) ^ (0xBF58476D1CE4E5B9LL * v33);
            v132 = v33;
            sub_A32210((__int64)&v139, v142);
            if ( !(_DWORD)v142 )
              goto LABEL_271;
            v112 = v142 - 1;
            LODWORD(v113) = (v142 - 1) & v130;
            v42 = v141 + 1;
            v33 = v132;
            v38 = (_QWORD *)(v140 + 8LL * (unsigned int)v113);
            v114 = *v38;
            if ( v132 != *v38 )
            {
              v9 = v140 + 8LL * (v112 & (unsigned int)v130);
              v115 = 1;
              v38 = 0;
              while ( v114 != -1 )
              {
                if ( !v38 && v114 == -2 )
                  v38 = (_QWORD *)v9;
                v113 = v112 & (unsigned int)(v113 + v115);
                v9 = v140 + 8 * v113;
                v114 = *(_QWORD *)v9;
                if ( v132 == *(_QWORD *)v9 )
                  goto LABEL_255;
                ++v115;
              }
              goto LABEL_243;
            }
          }
          goto LABEL_85;
        }
LABEL_247:
        v133 = v33;
        sub_A32210((__int64)&v139, 2 * v142);
        if ( !(_DWORD)v142 )
        {
LABEL_271:
          LODWORD(v141) = v141 + 1;
          BUG();
        }
        v33 = v133;
        v116 = v142 - 1;
        v9 = (unsigned int)v141;
        LODWORD(v117) = (v142 - 1) & (((0xBF58476D1CE4E5B9LL * v133) >> 31) ^ (484763065 * v133));
        v42 = v141 + 1;
        v38 = (_QWORD *)(v140 + 8LL * (unsigned int)v117);
        v118 = *v38;
        if ( v133 != *v38 )
        {
          v9 = v140 + 8LL * (v116 & ((unsigned int)((0xBF58476D1CE4E5B9LL * v133) >> 31) ^ (484763065 * (_DWORD)v133)));
          v119 = 1;
          v38 = 0;
          while ( v118 != -1 )
          {
            if ( v118 == -2 && !v38 )
              v38 = (_QWORD *)v9;
            v117 = v116 & (unsigned int)(v117 + v119);
            v9 = v140 + 8 * v117;
            v118 = *(_QWORD *)v9;
            if ( v133 == *(_QWORD *)v9 )
            {
LABEL_255:
              v38 = (_QWORD *)v9;
              goto LABEL_85;
            }
            ++v119;
          }
LABEL_243:
          if ( !v38 )
            v38 = (_QWORD *)v9;
        }
LABEL_85:
        LODWORD(v141) = v42;
        if ( *v38 != -1 )
          --HIDWORD(v141);
        *v38 = v33;
      }
LABEL_65:
      v31 = (__int64 *)v31[1];
      if ( a3 + 3 == v31 )
      {
        v3 = a1;
        if ( (v154 & 2) == 0 )
          goto LABEL_99;
LABEL_96:
        sub_25CE1D0(&v143, v9);
      }
    }
    ++v139;
    goto LABEL_247;
  }
  v29 = *a3;
  (*(void (__fastcall **)(__m128i **))(*(_QWORD *)v137[1] + 32LL))(&v172);
  v30 = (__m128i *)sub_2241130((unsigned __int64 *)&v172, 0, 0, "could not open contextual profile file: ", 0x28u);
  v186 = &v188;
  if ( (__m128i *)v30->m128i_i64[0] == &v30[1] )
  {
    v188 = _mm_loadu_si128(v30 + 1);
  }
  else
  {
    v186 = (__m128i *)v30->m128i_i64[0];
    v188.m128i_i64[0] = v30[1].m128i_i64[0];
  }
  v187 = v30->m128i_u64[1];
  v30->m128i_i64[0] = (__int64)v30[1].m128i_i64;
  v30->m128i_i64[1] = 0;
  v30[1].m128i_i8[0] = 0;
  LOWORD(v204) = 260;
  v200 = &v186;
  sub_B6ECE0(v29, (__int64)&v200);
  if ( v186 != &v188 )
    j_j___libc_free_0((unsigned __int64)v186);
  if ( v172 != &v174 )
    j_j___libc_free_0((unsigned __int64)v172);
  memset(a1, 0, 0x90u);
  a1[3] = a1 + 1;
  a1[4] = a1 + 1;
  a1[9] = a1 + 7;
  a1[10] = a1 + 7;
  a1[15] = a1 + 13;
  a1[16] = a1 + 13;
LABEL_59:
  if ( (v138 & 1) == 0 && v137[0] )
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v137[0] + 8LL))(v137[0]);
  return v3;
}

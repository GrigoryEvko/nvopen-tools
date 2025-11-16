// Function: sub_2092EF0
// Address: 0x2092ef0
//
void __fastcall sub_2092EF0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __m128i *v6; // r14
  unsigned __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rbx
  unsigned int v11; // esi
  __int64 v12; // r11
  unsigned int v13; // r13d
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rbx
  __int64 *v18; // rax
  char v19; // dl
  _BYTE *v20; // r13
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // rax
  char v26; // di
  unsigned int v27; // esi
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // r15
  unsigned __int8 v32; // al
  __int64 v33; // rdi
  __int64 *v34; // rsi
  int v35; // ebx
  __int64 v36; // rax
  __int64 v37; // r12
  __int64 v38; // rax
  __int64 v39; // r15
  __int64 v40; // r12
  __int64 v41; // rax
  __int64 v42; // rcx
  __int64 v43; // rdx
  int v44; // eax
  int v45; // ecx
  __int64 v46; // rsi
  const __m128i *v47; // r12
  int v48; // r15d
  __int64 v49; // rdi
  _BYTE *v50; // rax
  __m128i *v51; // rsi
  char v52; // di
  unsigned int v53; // eax
  __int64 *v54; // rsi
  __int64 *v55; // rcx
  int v56; // eax
  __int64 v57; // rdx
  _QWORD *v58; // rax
  _QWORD *i; // rdx
  unsigned int v60; // ecx
  _QWORD *v61; // rdi
  unsigned int v62; // eax
  __int64 v63; // rax
  unsigned __int64 v64; // rax
  unsigned __int64 v65; // rax
  int v66; // ebx
  __int64 v67; // r12
  _QWORD *v68; // rax
  __int64 v69; // rdx
  _QWORD *j; // rdx
  unsigned int v71; // esi
  __int64 v72; // r11
  unsigned int v73; // edi
  __int64 *v74; // r12
  __int64 v75; // rcx
  int v76; // eax
  __int64 v77; // r8
  unsigned int v78; // ecx
  __int64 v79; // rsi
  unsigned int v80; // edx
  __int64 *v81; // rax
  __int64 v82; // rdi
  __int64 v83; // rax
  int v84; // r10d
  __int64 *v85; // rcx
  int v86; // edx
  int v87; // edx
  int v88; // r11d
  int v89; // r11d
  __int64 v90; // r10
  __int64 v91; // rcx
  __int64 v92; // r8
  int v93; // edi
  __int64 *v94; // rsi
  int v95; // r11d
  int v96; // r11d
  __int64 v97; // r10
  __int64 v98; // rcx
  int v99; // edi
  __int64 v100; // r8
  int v101; // r11d
  int v102; // r11d
  __int64 v103; // r8
  unsigned int v104; // edx
  int v105; // eax
  __int64 v106; // rdi
  int v107; // esi
  __int64 *v108; // rcx
  _QWORD *v109; // rax
  int v110; // edx
  __int64 *v111; // rax
  int v112; // eax
  int v113; // r11d
  int v114; // r11d
  __int64 v115; // r8
  int v116; // esi
  unsigned int v117; // edx
  __int64 v118; // rdi
  int v119; // eax
  int v120; // r10d
  __int64 v121; // [rsp+10h] [rbp-1B0h]
  __m128i *v123; // [rsp+20h] [rbp-1A0h]
  __int64 v124; // [rsp+28h] [rbp-198h]
  __int64 v125; // [rsp+30h] [rbp-190h]
  __int64 v126; // [rsp+30h] [rbp-190h]
  __int64 v127; // [rsp+30h] [rbp-190h]
  __int64 v128; // [rsp+38h] [rbp-188h]
  __int64 v129; // [rsp+40h] [rbp-180h]
  __int64 v130; // [rsp+48h] [rbp-178h]
  __int64 v131; // [rsp+48h] [rbp-178h]
  __int64 v132; // [rsp+58h] [rbp-168h]
  __int64 v133; // [rsp+60h] [rbp-160h]
  __int64 v134; // [rsp+68h] [rbp-158h]
  __int64 v135; // [rsp+68h] [rbp-158h]
  int v136; // [rsp+68h] [rbp-158h]
  unsigned int v137; // [rsp+68h] [rbp-158h]
  __int64 v138; // [rsp+70h] [rbp-150h]
  int v139; // [rsp+78h] [rbp-148h]
  unsigned int v140; // [rsp+7Ch] [rbp-144h]
  char v141; // [rsp+8Bh] [rbp-135h] BYREF
  unsigned int v142; // [rsp+8Ch] [rbp-134h] BYREF
  __int64 v143; // [rsp+90h] [rbp-130h] BYREF
  __int64 v144; // [rsp+98h] [rbp-128h]
  __int64 v145; // [rsp+A0h] [rbp-120h] BYREF
  __int64 v146; // [rsp+A8h] [rbp-118h]
  __int64 v147; // [rsp+B0h] [rbp-110h] BYREF
  __int64 v148; // [rsp+B8h] [rbp-108h]
  __int64 v149; // [rsp+C0h] [rbp-100h] BYREF
  __int64 v150; // [rsp+C8h] [rbp-F8h]
  __m128i v151; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v152; // [rsp+E0h] [rbp-E0h]
  __int64 v153; // [rsp+F0h] [rbp-D0h] BYREF
  __int64 *v154; // [rsp+F8h] [rbp-C8h]
  __int64 *v155; // [rsp+100h] [rbp-C0h]
  __int64 v156; // [rsp+108h] [rbp-B8h]
  int v157; // [rsp+110h] [rbp-B0h]
  _BYTE v158[40]; // [rsp+118h] [rbp-A8h] BYREF
  _BYTE *v159; // [rsp+140h] [rbp-80h] BYREF
  __int64 v160; // [rsp+148h] [rbp-78h]
  _BYTE v161[112]; // [rsp+150h] [rbp-70h] BYREF

  v6 = &v151;
  v7 = sub_157EBA0(a2);
  v153 = 0;
  v129 = v7;
  v154 = (__int64 *)v158;
  v155 = (__int64 *)v158;
  v156 = 4;
  v157 = 0;
  v140 = 0;
  v139 = sub_15F4D60(v7);
  if ( v139 )
  {
    while ( 1 )
    {
      v8 = sub_15F4DF0(v129, v140);
      v9 = *(_QWORD *)(v8 + 48);
      if ( !v9 )
LABEL_195:
        BUG();
      if ( *(_BYTE *)(v9 - 8) == 77 )
        break;
LABEL_3:
      if ( v139 == ++v140 )
        goto LABEL_66;
    }
    v10 = *(_QWORD *)(a1 + 712);
    v11 = *(_DWORD *)(v10 + 72);
    if ( v11 )
    {
      v12 = *(_QWORD *)(v10 + 56);
      v13 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
      v14 = (v11 - 1) & v13;
      v15 = (__int64 *)(v12 + 16LL * v14);
      v16 = *v15;
      if ( v8 == *v15 )
      {
LABEL_8:
        v17 = v15[1];
        goto LABEL_9;
      }
      v84 = 1;
      v85 = 0;
      while ( v16 != -8 )
      {
        if ( v16 == -16 && !v85 )
          v85 = v15;
        v14 = (v11 - 1) & (v84 + v14);
        v15 = (__int64 *)(v12 + 16LL * v14);
        v16 = *v15;
        if ( v8 == *v15 )
          goto LABEL_8;
        ++v84;
      }
      v86 = *(_DWORD *)(v10 + 64);
      if ( v85 )
        v15 = v85;
      ++*(_QWORD *)(v10 + 48);
      v87 = v86 + 1;
      if ( 4 * v87 < 3 * v11 )
      {
        if ( v11 - *(_DWORD *)(v10 + 68) - v87 > v11 >> 3 )
        {
LABEL_116:
          *(_DWORD *)(v10 + 64) = v87;
          if ( *v15 != -8 )
            --*(_DWORD *)(v10 + 68);
          *v15 = v8;
          v17 = 0;
          v15[1] = 0;
LABEL_9:
          v18 = v154;
          if ( v155 != v154 )
            goto LABEL_10;
          v54 = &v154[HIDWORD(v156)];
          if ( v154 == v54 )
            goto LABEL_107;
          v55 = 0;
          do
          {
            if ( *v18 == v17 )
              goto LABEL_3;
            if ( *v18 == -2 )
              v55 = v18;
            ++v18;
          }
          while ( v54 != v18 );
          if ( !v55 )
          {
LABEL_107:
            if ( HIDWORD(v156) >= (unsigned int)v156 )
            {
LABEL_10:
              sub_16CCBA0((__int64)&v153, v17);
              if ( !v19 )
                goto LABEL_3;
              goto LABEL_11;
            }
            ++HIDWORD(v156);
            *v54 = v17;
            ++v153;
          }
          else
          {
            *v55 = v17;
            --v157;
            ++v153;
          }
LABEL_11:
          v20 = *(_BYTE **)(v17 + 32);
          v21 = sub_157F280(v8);
          v128 = v22;
          v138 = v21;
          if ( v21 == v22 )
            goto LABEL_3;
          v123 = v6;
          v23 = a1;
          while ( !*(_QWORD *)(v138 + 8) || (unsigned __int8)sub_1642FB0(*(_QWORD *)v138) )
          {
LABEL_13:
            v24 = *(_QWORD *)(v138 + 32);
            if ( !v24 )
              goto LABEL_195;
            v138 = 0;
            if ( *(_BYTE *)(v24 - 8) == 77 )
              v138 = v24 - 24;
            if ( v128 == v138 )
            {
              a1 = v23;
              v6 = v123;
              goto LABEL_3;
            }
          }
          v25 = 0x17FFFFFFE8LL;
          v26 = *(_BYTE *)(v138 + 23) & 0x40;
          v27 = *(_DWORD *)(v138 + 20) & 0xFFFFFFF;
          if ( v27 )
          {
            v28 = 24LL * *(unsigned int *)(v138 + 56) + 8;
            v29 = 0;
            do
            {
              v30 = v138 - 24LL * v27;
              if ( v26 )
                v30 = *(_QWORD *)(v138 - 8);
              if ( a2 == *(_QWORD *)(v30 + v28) )
              {
                v25 = 24 * v29;
                goto LABEL_26;
              }
              ++v29;
              v28 += 8;
            }
            while ( v27 != (_DWORD)v29 );
            v25 = 0x17FFFFFFE8LL;
          }
LABEL_26:
          if ( v26 )
          {
            v31 = *(_QWORD *)(*(_QWORD *)(v138 - 8) + v25);
            if ( !v31 )
              goto LABEL_76;
          }
          else
          {
            v31 = *(_QWORD *)(v138 - 24LL * v27 + v25);
            if ( !v31 )
LABEL_76:
              BUG();
          }
          v32 = *(_BYTE *)(v31 + 16);
          if ( v32 > 0x10u )
          {
            v77 = *(_QWORD *)(v23 + 712);
            v78 = *(_DWORD *)(v77 + 232);
            if ( v78 )
            {
              v79 = *(_QWORD *)(v77 + 216);
              v80 = (v78 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
              v81 = (__int64 *)(v79 + 16LL * v80);
              v82 = *v81;
              if ( *v81 == v31 )
              {
LABEL_103:
                if ( v81 != (__int64 *)(v79 + 16LL * v78) )
                {
                  v35 = *((_DWORD *)v81 + 2);
                  goto LABEL_32;
                }
              }
              else
              {
                v119 = 1;
                while ( v82 != -8 )
                {
                  v120 = v119 + 1;
                  v80 = (v78 - 1) & (v119 + v80);
                  v81 = (__int64 *)(v79 + 16LL * v80);
                  v82 = *v81;
                  if ( v31 == *v81 )
                    goto LABEL_103;
                  v119 = v120;
                }
              }
            }
            v34 = *(__int64 **)v31;
            v33 = *(_QWORD *)(v23 + 712);
            goto LABEL_31;
          }
          if ( v32 != 9 )
          {
            v33 = *(_QWORD *)(v23 + 712);
            v34 = *(__int64 **)v31;
LABEL_31:
            v35 = sub_1FDE000(v33, v34);
            sub_208C270(v23, (__int64 *)v31, v35, a3, a4, a5);
LABEL_32:
            v159 = v161;
            v160 = 0x400000000LL;
            v36 = *(_QWORD *)(v23 + 552);
            v37 = *(_QWORD *)v138;
            v133 = *(_QWORD *)(v36 + 16);
            v38 = sub_1E0A0C0(*(_QWORD *)(v36 + 32));
            sub_20C7CE0(v133, v38, v37, &v159, 0, 0);
            if ( (_DWORD)v160 )
            {
              v39 = (__int64)v123;
              v40 = 0;
              v132 = 16LL * (unsigned int)v160;
              do
              {
                v41 = (unsigned __int8)v159[v40];
                v42 = *(_QWORD *)&v159[v40 + 8];
                v43 = *(_QWORD *)(v23 + 552);
                LOBYTE(v143) = v41;
                v144 = v42;
                if ( (_BYTE)v41 )
                {
                  v44 = *(unsigned __int8 *)(v133 + v41 + 1040);
                }
                else
                {
                  v135 = *(_QWORD *)(v43 + 48);
                  v130 = v42;
                  if ( sub_1F58D20((__int64)&v143) )
                  {
                    v151.m128i_i8[0] = 0;
                    v151.m128i_i64[1] = 0;
                    LOBYTE(v147) = 0;
                    v44 = sub_1F426C0(v133, v135, (unsigned int)v143, v130, v39, (unsigned int *)&v149, &v147);
                  }
                  else
                  {
                    v124 = v135;
                    v136 = sub_1F58D40((__int64)&v143);
                    v145 = v143;
                    v125 = v143;
                    v146 = v144;
                    v131 = v144;
                    if ( sub_1F58D20((__int64)&v145) )
                    {
                      v151.m128i_i8[0] = 0;
                      v151.m128i_i64[1] = 0;
                      LOBYTE(v147) = 0;
                      sub_1F426C0(v133, v124, (unsigned int)v145, v146, v39, (unsigned int *)&v149, &v147);
                      v52 = v147;
                    }
                    else
                    {
                      sub_1F40D10(v39, v133, v124, v125, v131);
                      LOBYTE(v147) = v151.m128i_i8[8];
                      v148 = v152;
                      if ( v151.m128i_i8[8] )
                      {
                        v52 = *(_BYTE *)(v133 + v151.m128i_u8[8] + 1155);
                      }
                      else
                      {
                        v126 = v152;
                        if ( sub_1F58D20((__int64)&v147) )
                        {
                          v151.m128i_i8[0] = 0;
                          v151.m128i_i64[1] = 0;
                          LOBYTE(v142) = 0;
                          sub_1F426C0(v133, v124, (unsigned int)v147, v126, v39, (unsigned int *)&v149, &v142);
                          v52 = v142;
                        }
                        else
                        {
                          sub_1F40D10(v39, v133, v124, v147, v148);
                          LOBYTE(v149) = v151.m128i_i8[8];
                          v150 = v152;
                          if ( v151.m128i_i8[8] )
                          {
                            v52 = *(_BYTE *)(v133 + v151.m128i_u8[8] + 1155);
                          }
                          else
                          {
                            v127 = v152;
                            if ( sub_1F58D20((__int64)&v149) )
                            {
                              v151.m128i_i8[0] = 0;
                              v151.m128i_i64[1] = 0;
                              v141 = 0;
                              sub_1F426C0(v133, v124, (unsigned int)v149, v127, v39, &v142, &v141);
                              v52 = v141;
                            }
                            else
                            {
                              sub_1F40D10(v39, v133, v124, v149, v150);
                              v83 = v121;
                              LOBYTE(v83) = v151.m128i_i8[8];
                              v121 = v83;
                              v52 = sub_1D5E9F0(v133, v124, (unsigned int)v83, v152);
                            }
                          }
                        }
                      }
                    }
                    v53 = sub_2045180(v52);
                    v44 = (v53 + v136 - 1) / v53;
                  }
                }
                v45 = v35;
                if ( v44 )
                {
                  v134 = v40;
                  v46 = (__int64)v20;
                  v47 = (const __m128i *)v39;
                  v48 = v35 + v44;
                  while ( 1 )
                  {
                    v49 = *(_QWORD *)(v23 + 712);
                    if ( !v20 )
                      BUG();
                    v50 = v20;
                    if ( (*v20 & 4) == 0 && (v20[46] & 8) != 0 )
                    {
                      do
                        v50 = (_BYTE *)*((_QWORD *)v50 + 1);
                      while ( (v50[46] & 8) != 0 );
                    }
                    v20 = (_BYTE *)*((_QWORD *)v50 + 1);
                    v151.m128i_i64[0] = v46;
                    v151.m128i_i32[2] = v35;
                    v51 = *(__m128i **)(v49 + 912);
                    if ( v51 == *(__m128i **)(v49 + 920) )
                    {
                      ++v35;
                      sub_1FD42F0((const __m128i **)(v49 + 904), v51, v47);
                      if ( v48 == v35 )
                        goto LABEL_48;
                    }
                    else
                    {
                      if ( v51 )
                      {
                        a3 = _mm_loadu_si128(&v151);
                        *v51 = a3;
                        v51 = *(__m128i **)(v49 + 912);
                      }
                      ++v35;
                      *(_QWORD *)(v49 + 912) = v51 + 1;
                      if ( v48 == v35 )
                      {
LABEL_48:
                        v45 = v48;
                        v39 = (__int64)v47;
                        v40 = v134;
                        break;
                      }
                    }
                    v46 = (__int64)v20;
                  }
                }
                v35 = v45;
                v40 += 16;
              }
              while ( v132 != v40 );
            }
            if ( v159 != v161 )
              _libc_free((unsigned __int64)v159);
            goto LABEL_13;
          }
          v71 = *(_DWORD *)(v23 + 704);
          if ( v71 )
          {
            v72 = *(_QWORD *)(v23 + 688);
            v73 = (v71 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
            v74 = (__int64 *)(v72 + 16LL * v73);
            v75 = *v74;
            if ( v31 == *v74 )
            {
LABEL_98:
              v35 = *((_DWORD *)v74 + 2);
              if ( v35 )
                goto LABEL_32;
LABEL_99:
              v76 = sub_1FDE000(*(_QWORD *)(v23 + 712), *(__int64 **)v31);
              *((_DWORD *)v74 + 2) = v76;
              sub_208C270(v23, (__int64 *)v31, v76, a3, a4, a5);
              v35 = *((_DWORD *)v74 + 2);
              goto LABEL_32;
            }
            v110 = 1;
            v111 = 0;
            while ( v75 != -8 )
            {
              if ( v75 == -16 && !v111 )
                v111 = v74;
              v73 = (v71 - 1) & (v110 + v73);
              v74 = (__int64 *)(v72 + 16LL * v73);
              v75 = *v74;
              if ( v31 == *v74 )
                goto LABEL_98;
              ++v110;
            }
            if ( v111 )
              v74 = v111;
            v112 = *(_DWORD *)(v23 + 696);
            ++*(_QWORD *)(v23 + 680);
            v105 = v112 + 1;
            if ( 4 * v105 < 3 * v71 )
            {
              if ( v71 - *(_DWORD *)(v23 + 700) - v105 > v71 >> 3 )
                goto LABEL_140;
              v137 = ((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4);
              sub_2063810(v23 + 680, v71);
              v113 = *(_DWORD *)(v23 + 704);
              if ( !v113 )
              {
LABEL_193:
                ++*(_DWORD *)(v23 + 696);
                BUG();
              }
              v114 = v113 - 1;
              v108 = 0;
              v115 = *(_QWORD *)(v23 + 688);
              v116 = 1;
              v117 = v114 & v137;
              v105 = *(_DWORD *)(v23 + 696) + 1;
              v74 = (__int64 *)(v115 + 16LL * (v114 & v137));
              v118 = *v74;
              if ( v31 == *v74 )
                goto LABEL_140;
              while ( v118 != -8 )
              {
                if ( !v108 && v118 == -16 )
                  v108 = v74;
                v117 = v114 & (v116 + v117);
                v74 = (__int64 *)(v115 + 16LL * v117);
                v118 = *v74;
                if ( v31 == *v74 )
                  goto LABEL_140;
                ++v116;
              }
              goto LABEL_145;
            }
          }
          else
          {
            ++*(_QWORD *)(v23 + 680);
          }
          sub_2063810(v23 + 680, 2 * v71);
          v101 = *(_DWORD *)(v23 + 704);
          if ( !v101 )
            goto LABEL_193;
          v102 = v101 - 1;
          v103 = *(_QWORD *)(v23 + 688);
          v104 = v102 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
          v105 = *(_DWORD *)(v23 + 696) + 1;
          v74 = (__int64 *)(v103 + 16LL * v104);
          v106 = *v74;
          if ( v31 == *v74 )
            goto LABEL_140;
          v107 = 1;
          v108 = 0;
          while ( v106 != -8 )
          {
            if ( !v108 && v106 == -16 )
              v108 = v74;
            v104 = v102 & (v107 + v104);
            v74 = (__int64 *)(v103 + 16LL * v104);
            v106 = *v74;
            if ( v31 == *v74 )
              goto LABEL_140;
            ++v107;
          }
LABEL_145:
          if ( v108 )
            v74 = v108;
LABEL_140:
          *(_DWORD *)(v23 + 696) = v105;
          if ( *v74 != -8 )
            --*(_DWORD *)(v23 + 700);
          *v74 = v31;
          *((_DWORD *)v74 + 2) = 0;
          goto LABEL_99;
        }
        sub_1D52F30(v10 + 48, v11);
        v95 = *(_DWORD *)(v10 + 72);
        if ( v95 )
        {
          v96 = v95 - 1;
          v97 = *(_QWORD *)(v10 + 56);
          LODWORD(v98) = v96 & v13;
          v99 = 1;
          v94 = 0;
          v87 = *(_DWORD *)(v10 + 64) + 1;
          v15 = (__int64 *)(v97 + 16LL * (v96 & v13));
          v100 = *v15;
          if ( v8 == *v15 )
            goto LABEL_116;
          while ( v100 != -8 )
          {
            if ( v100 == -16 && !v94 )
              v94 = v15;
            v98 = v96 & (unsigned int)(v98 + v99);
            v15 = (__int64 *)(v97 + 16 * v98);
            v100 = *v15;
            if ( v8 == *v15 )
              goto LABEL_116;
            ++v99;
          }
          goto LABEL_132;
        }
        goto LABEL_196;
      }
    }
    else
    {
      ++*(_QWORD *)(v10 + 48);
    }
    sub_1D52F30(v10 + 48, 2 * v11);
    v88 = *(_DWORD *)(v10 + 72);
    if ( v88 )
    {
      v89 = v88 - 1;
      v90 = *(_QWORD *)(v10 + 56);
      LODWORD(v91) = v89 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v87 = *(_DWORD *)(v10 + 64) + 1;
      v15 = (__int64 *)(v90 + 16LL * (unsigned int)v91);
      v92 = *v15;
      if ( v8 == *v15 )
        goto LABEL_116;
      v93 = 1;
      v94 = 0;
      while ( v92 != -8 )
      {
        if ( !v94 && v92 == -16 )
          v94 = v15;
        v91 = v89 & (unsigned int)(v91 + v93);
        v15 = (__int64 *)(v90 + 16 * v91);
        v92 = *v15;
        if ( v8 == *v15 )
          goto LABEL_116;
        ++v93;
      }
LABEL_132:
      if ( v94 )
        v15 = v94;
      goto LABEL_116;
    }
LABEL_196:
    ++*(_DWORD *)(v10 + 64);
    BUG();
  }
LABEL_66:
  v56 = *(_DWORD *)(a1 + 696);
  ++*(_QWORD *)(a1 + 680);
  if ( !v56 )
  {
    if ( !*(_DWORD *)(a1 + 700) )
      goto LABEL_72;
    v57 = *(unsigned int *)(a1 + 704);
    if ( (unsigned int)v57 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 688));
      *(_QWORD *)(a1 + 688) = 0;
      *(_QWORD *)(a1 + 696) = 0;
      *(_DWORD *)(a1 + 704) = 0;
      goto LABEL_72;
    }
    goto LABEL_69;
  }
  v60 = 4 * v56;
  v57 = *(unsigned int *)(a1 + 704);
  if ( (unsigned int)(4 * v56) < 0x40 )
    v60 = 64;
  if ( v60 >= (unsigned int)v57 )
  {
LABEL_69:
    v58 = *(_QWORD **)(a1 + 688);
    for ( i = &v58[2 * v57]; i != v58; v58 += 2 )
      *v58 = -8;
    *(_QWORD *)(a1 + 696) = 0;
    goto LABEL_72;
  }
  v61 = *(_QWORD **)(a1 + 688);
  v62 = v56 - 1;
  if ( !v62 )
  {
    v67 = 2048;
    v66 = 128;
LABEL_91:
    j___libc_free_0(v61);
    *(_DWORD *)(a1 + 704) = v66;
    v68 = (_QWORD *)sub_22077B0(v67);
    v69 = *(unsigned int *)(a1 + 704);
    *(_QWORD *)(a1 + 696) = 0;
    *(_QWORD *)(a1 + 688) = v68;
    for ( j = &v68[2 * v69]; j != v68; v68 += 2 )
    {
      if ( v68 )
        *v68 = -8;
    }
    goto LABEL_72;
  }
  _BitScanReverse(&v62, v62);
  v63 = (unsigned int)(1 << (33 - (v62 ^ 0x1F)));
  if ( (int)v63 < 64 )
    v63 = 64;
  if ( (_DWORD)v63 != (_DWORD)v57 )
  {
    v64 = (4 * (int)v63 / 3u + 1) | ((unsigned __int64)(4 * (int)v63 / 3u + 1) >> 1);
    v65 = ((v64 | (v64 >> 2)) >> 4) | v64 | (v64 >> 2) | ((((v64 | (v64 >> 2)) >> 4) | v64 | (v64 >> 2)) >> 8);
    v66 = (v65 | (v65 >> 16)) + 1;
    v67 = 16 * ((v65 | (v65 >> 16)) + 1);
    goto LABEL_91;
  }
  *(_QWORD *)(a1 + 696) = 0;
  v109 = &v61[2 * v63];
  do
  {
    if ( v61 )
      *v61 = -8;
    v61 += 2;
  }
  while ( v109 != v61 );
LABEL_72:
  if ( v155 != v154 )
    _libc_free((unsigned __int64)v155);
}

// Function: sub_F24210
// Address: 0xf24210
//
__int64 __fastcall sub_F24210(_QWORD *a1, __int64 a2)
{
  bool v2; // zf
  __int64 *v3; // r14
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  _BYTE *v8; // rdi
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // r15
  unsigned __int8 *v13; // r8
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 v17; // rbx
  __int64 v18; // r15
  __int64 v19; // r13
  __int64 v20; // rsi
  __int64 v21; // rbx
  __int64 v22; // r14
  __int64 v23; // rbx
  __int64 v24; // r14
  __int64 v25; // rdi
  __int64 v26; // rsi
  __int64 v27; // rbx
  __int64 v28; // r13
  __int64 v29; // rdi
  __int64 v30; // rdi
  __int64 v31; // rbx
  __int64 v32; // r13
  __int64 v33; // rbx
  __int64 v34; // r13
  unsigned __int64 *v35; // rbx
  unsigned __int64 *v36; // r13
  __int64 v37; // rax
  __int64 v39; // rdx
  int v40; // eax
  unsigned __int64 *v41; // rdi
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // r13
  __int64 v45; // rbx
  __int64 *v46; // r12
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rdx
  __int64 v50; // rdi
  __int64 v51; // rsi
  __int64 v52; // r14
  __int64 *v53; // r12
  __int64 v54; // rdx
  __int64 v55; // rdi
  __int64 *v56; // rsi
  unsigned __int64 *v57; // r12
  unsigned __int64 v58; // rax
  __int64 v59; // rbx
  unsigned __int8 v60; // r12
  _QWORD *v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // r13
  __int64 *v65; // r12
  __int64 v66; // rbx
  __int64 *v67; // r13
  __int64 v68; // rdi
  __int64 v69; // rax
  _BYTE *v70; // r12
  _BYTE *v71; // rbx
  __int64 v72; // rdi
  __int64 v73; // rax
  int v74; // eax
  unsigned __int64 *v75; // rdi
  unsigned __int8 *v76; // rdx
  char v77; // al
  __int64 v78; // rcx
  __int64 v79; // rdx
  unsigned __int8 *v80; // rax
  __int64 *v81; // r12
  __int64 v82; // rax
  __int64 v83; // rax
  unsigned int v84; // eax
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 v89; // rax
  unsigned __int8 *v90; // rdi
  __int64 v91; // rdx
  __int64 v92; // rdx
  __int64 v93; // rdi
  __int64 v94; // rdx
  __int64 v95; // rax
  int v96; // eax
  __int64 v97; // rax
  unsigned __int64 *v98; // rdi
  unsigned __int8 *v99; // rax
  __int32 v100; // eax
  __int64 v101; // r14
  __int64 *v102; // rax
  __int64 v103; // r12
  __int64 v104; // rbx
  __int64 v105; // r13
  _QWORD *v106; // rdi
  __int64 *v107; // r14
  __int64 *v108; // rbx
  __int64 v109; // r13
  __int64 v110; // rax
  _BYTE *v111; // r12
  _BYTE *v112; // rbx
  _QWORD *v113; // r13
  __int64 v114; // rax
  __int64 v115; // rax
  unsigned __int64 *v116; // rdx
  unsigned __int64 *v117; // rdi
  unsigned __int8 *v118; // rax
  __int32 v119; // eax
  unsigned __int64 *v120; // rdx
  unsigned __int64 *v121; // rax
  unsigned __int64 *v122; // rdi
  unsigned __int8 *v123; // rdx
  __int32 v124; // edx
  unsigned __int64 *v125; // rax
  char v126; // al
  char v127; // al
  unsigned __int8 *v128; // r8
  char v129; // al
  __int64 v130; // rdx
  __int64 v131; // rcx
  __int64 v132; // r8
  __int64 v133; // r9
  unsigned __int64 *v134; // rdi
  unsigned __int8 *v135; // rax
  __int32 v136; // eax
  _BYTE *v137; // rax
  _BYTE *v138; // rax
  int v139; // eax
  unsigned __int8 *v140; // [rsp+8h] [rbp-7E8h]
  int v141; // [rsp+10h] [rbp-7E0h]
  __int32 v142; // [rsp+10h] [rbp-7E0h]
  __int32 v143; // [rsp+10h] [rbp-7E0h]
  __int32 v144; // [rsp+10h] [rbp-7E0h]
  __int32 v145; // [rsp+10h] [rbp-7E0h]
  unsigned __int8 *v146; // [rsp+18h] [rbp-7D8h]
  unsigned __int8 *v147; // [rsp+18h] [rbp-7D8h]
  __int32 v148; // [rsp+18h] [rbp-7D8h]
  __int64 v149; // [rsp+18h] [rbp-7D8h]
  unsigned __int64 *v150; // [rsp+18h] [rbp-7D8h]
  unsigned __int64 *v151; // [rsp+18h] [rbp-7D8h]
  unsigned __int64 *v152; // [rsp+18h] [rbp-7D8h]
  unsigned __int64 *v153; // [rsp+18h] [rbp-7D8h]
  unsigned __int64 *v154; // [rsp+18h] [rbp-7D8h]
  __int64 v155; // [rsp+18h] [rbp-7D8h]
  unsigned __int64 *v156; // [rsp+18h] [rbp-7D8h]
  __int64 v157; // [rsp+18h] [rbp-7D8h]
  __int64 v158; // [rsp+18h] [rbp-7D8h]
  char v160; // [rsp+28h] [rbp-7C8h]
  __int64 *v161; // [rsp+28h] [rbp-7C8h]
  __int64 v162; // [rsp+48h] [rbp-7A8h]
  __int64 v163; // [rsp+48h] [rbp-7A8h]
  __int64 v164; // [rsp+48h] [rbp-7A8h]
  unsigned __int8 *v165; // [rsp+58h] [rbp-798h]
  __int64 v166; // [rsp+60h] [rbp-790h]
  __int64 v167; // [rsp+60h] [rbp-790h]
  unsigned __int16 v168; // [rsp+60h] [rbp-790h]
  __int64 v169; // [rsp+68h] [rbp-788h]
  unsigned int v170; // [rsp+74h] [rbp-77Ch] BYREF
  unsigned __int8 *v171; // [rsp+78h] [rbp-778h] BYREF
  void *s2; // [rsp+80h] [rbp-770h] BYREF
  __int64 v173; // [rsp+88h] [rbp-768h]
  char v174; // [rsp+90h] [rbp-760h]
  _BYTE *v175; // [rsp+A0h] [rbp-750h] BYREF
  __int64 v176; // [rsp+A8h] [rbp-748h]
  _BYTE v177[32]; // [rsp+B0h] [rbp-740h] BYREF
  __m128i s1; // [rsp+D0h] [rbp-720h] BYREF
  _BYTE v179[16]; // [rsp+E0h] [rbp-710h] BYREF
  __int16 v180; // [rsp+F0h] [rbp-700h]
  char v181; // [rsp+100h] [rbp-6F0h]
  __int64 *v182; // [rsp+110h] [rbp-6E0h] BYREF
  __int64 v183; // [rsp+118h] [rbp-6D8h]
  _BYTE v184[64]; // [rsp+120h] [rbp-6D0h] BYREF
  _BYTE *v185; // [rsp+160h] [rbp-690h] BYREF
  __int64 v186; // [rsp+168h] [rbp-688h]
  _BYTE v187[64]; // [rsp+170h] [rbp-680h] BYREF
  unsigned __int64 *v188; // [rsp+1B0h] [rbp-640h] BYREF
  __int64 v189; // [rsp+1B8h] [rbp-638h]
  _BYTE v190[1584]; // [rsp+1C0h] [rbp-630h] BYREF

  v2 = *(_BYTE *)a2 == 60;
  v188 = (unsigned __int64 *)v190;
  v189 = 0x4000000000LL;
  v182 = (__int64 *)v184;
  v165 = (unsigned __int8 *)a2;
  v183 = 0x800000000LL;
  v185 = v187;
  v186 = 0x800000000LL;
  v169 = 0;
  if ( v2 )
  {
    sub_AE7A50((__int64)&v182, a2, (__int64)&v185);
    v81 = (__int64 *)sub_B43CA0(a2);
    v82 = sub_22077B0(432);
    v169 = v82;
    if ( v82 )
      sub_AE0470(v82, v81, 0, 0);
  }
  v3 = (__int64 *)a1[9];
  v175 = v177;
  v176 = 0x400000000LL;
  sub_D5D330((__int64)&s2, a2, v3);
  v160 = v174;
  v6 = (unsigned int)v176;
  v7 = (unsigned int)v176 + 1LL;
  if ( v7 > HIDWORD(v176) )
  {
    a2 = (__int64)v177;
    sub_C8D5F0((__int64)&v175, v177, v7, 8u, v4, v5);
    v6 = (unsigned int)v176;
  }
  *(_QWORD *)&v175[8 * v6] = v165;
  v8 = v175;
  v9 = v176 + 1;
  LODWORD(v176) = v176 + 1;
LABEL_5:
  v10 = v9--;
  v11 = *(_QWORD *)&v8[8 * v10 - 8];
  LODWORD(v176) = v9;
  v12 = *(_QWORD *)(v11 + 16);
  if ( !v12 )
    goto LABEL_90;
  while ( 2 )
  {
    v13 = *(unsigned __int8 **)(v12 + 24);
    v171 = v13;
    switch ( *v13 )
    {
      case '>':
        if ( (v13[2] & 1) == 0 )
        {
          v73 = *((_QWORD *)v13 - 4);
          if ( v73 )
          {
            if ( v11 == v73 )
              goto LABEL_140;
          }
        }
        goto LABEL_8;
      case '?':
      case 'N':
      case 'O':
        v39 = (unsigned int)v189;
        v40 = v189;
        if ( HIDWORD(v189) > (unsigned int)v189 )
          goto LABEL_79;
        v116 = (unsigned __int64 *)sub_C8D7D0(
                                     (__int64)&v188,
                                     (__int64)v190,
                                     0,
                                     0x18u,
                                     (unsigned __int64 *)&s1,
                                     (__int64)&v188);
        v117 = &v116[3 * (unsigned int)v189];
        if ( v117 )
        {
          v118 = v171;
          v117[1] = 0;
          *v117 = 6;
          v117[2] = (unsigned __int64)v118;
          if ( v118 + 4096 != 0 && v118 != 0 && v118 != (unsigned __int8 *)-8192LL )
          {
            v151 = v116;
            sub_BD73F0((__int64)v117);
            v116 = v151;
          }
        }
        a2 = (__int64)v116;
        v152 = v116;
        sub_F17F80((__int64)&v188, v116);
        v119 = s1.m128i_i32[0];
        v120 = v152;
        if ( v188 != (unsigned __int64 *)v190 )
        {
          v143 = s1.m128i_i32[0];
          _libc_free(v188, a2);
          v119 = v143;
          v120 = v152;
        }
        LODWORD(v189) = v189 + 1;
        v13 = v171;
        v188 = v120;
        HIDWORD(v189) = v119;
        goto LABEL_85;
      case 'R':
        if ( (*((_WORD *)v13 + 1) & 0x3Fu) - 32 > 1 )
          goto LABEL_8;
        v76 = *(unsigned __int8 **)&v13[32 * (*((_QWORD *)v13 - 8) != 0 && v11 == *((_QWORD *)v13 - 8)) - 64];
        if ( *v76 == 20 )
          goto LABEL_151;
        if ( *v76 == 61 )
        {
          if ( **((_BYTE **)v76 - 4) != 3 )
            goto LABEL_8;
        }
        else
        {
          a2 = (__int64)v3;
          v146 = *(unsigned __int8 **)&v13[32 * (*((_QWORD *)v13 - 8) != 0 && v11 == *((_QWORD *)v13 - 8)) - 64];
          v77 = sub_D5CC50(v76, v3);
          if ( v165 == v146 || !v77 )
            goto LABEL_8;
        }
LABEL_151:
        if ( (unsigned __int8)(*v165 - 34) > 0x33u )
          goto LABEL_161;
        v78 = 0x8000000000041LL;
        if ( !_bittest64(&v78, (unsigned int)*v165 - 34) )
          goto LABEL_161;
        a2 = *((_QWORD *)v165 - 4);
        if ( a2 )
        {
          if ( *(_BYTE *)a2 )
          {
            a2 = 0;
          }
          else if ( *(_QWORD *)(a2 + 24) != *((_QWORD *)v165 + 10) )
          {
            a2 = 0;
          }
        }
        if ( !sub_981210(*v3, a2, &v170) )
          goto LABEL_161;
        a2 = (unsigned __int64)v170 >> 6;
        if ( (v3[a2 + 1] & (1LL << v170)) != 0
          || (((int)*(unsigned __int8 *)(*v3 + (v170 >> 2)) >> (2 * (v170 & 3))) & 3) == 0
          || v170 != 166 )
        {
          goto LABEL_161;
        }
        v89 = *((_DWORD *)v165 + 1) & 0x7FFFFFF;
        v90 = *(unsigned __int8 **)&v165[-32 * v89];
        v91 = *v90;
        if ( (_BYTE)v91 == 17 )
        {
          v92 = (__int64)(v90 + 24);
        }
        else
        {
          if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v90 + 1) + 8LL) - 17 > 1 )
            goto LABEL_8;
          if ( (unsigned __int8)v91 > 0x15u )
            goto LABEL_8;
          a2 = 0;
          v137 = sub_AD7630((__int64)v90, 0, v91);
          if ( !v137 || *v137 != 17 )
            goto LABEL_8;
          v92 = (__int64)(v137 + 24);
          v89 = *((_DWORD *)v165 + 1) & 0x7FFFFFF;
        }
        v93 = *(_QWORD *)&v165[32 * (1 - v89)];
        a2 = v93 + 24;
        if ( *(_BYTE *)v93 != 17 )
        {
          v157 = v92;
          if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v93 + 8) + 8LL) - 17 > 1 )
            goto LABEL_8;
          if ( *(_BYTE *)v93 > 0x15u )
            goto LABEL_8;
          a2 = 0;
          v138 = sub_AD7630(v93, 0, v92);
          if ( !v138 || *v138 != 17 )
            goto LABEL_8;
          v92 = v157;
          a2 = (__int64)(v138 + 24);
        }
        if ( *(_DWORD *)(v92 + 8) > 0x40u )
        {
          v158 = v92;
          v139 = sub_C44630(v92);
          v92 = v158;
          if ( v139 != 1 )
            goto LABEL_8;
        }
        else if ( !*(_QWORD *)v92 || (*(_QWORD *)v92 & (*(_QWORD *)v92 - 1LL)) != 0 )
        {
          goto LABEL_8;
        }
        sub_C4B490((__int64)&s1, a2, v92);
        if ( s1.m128i_i32[2] <= 0x40u )
        {
          if ( s1.m128i_i64[0] )
            goto LABEL_8;
LABEL_161:
          v79 = (unsigned int)v189;
          v74 = v189;
          if ( HIDWORD(v189) <= (unsigned int)v189 )
          {
            v121 = (unsigned __int64 *)sub_C8D7D0(
                                         (__int64)&v188,
                                         (__int64)v190,
                                         0,
                                         0x18u,
                                         (unsigned __int64 *)&s1,
                                         (__int64)&v188);
            v122 = &v121[3 * (unsigned int)v189];
            if ( v122 )
            {
              v123 = v171;
              v122[1] = 0;
              *v122 = 6;
              v122[2] = (unsigned __int64)v123;
              if ( v123 != 0 && v123 + 4096 != 0 && v123 != (unsigned __int8 *)-8192LL )
              {
                v153 = v121;
                sub_BD73F0((__int64)v122);
                v121 = v153;
              }
            }
            a2 = (__int64)v121;
            v154 = v121;
            sub_F17F80((__int64)&v188, v121);
            v124 = s1.m128i_i32[0];
            v125 = v154;
            if ( v188 != (unsigned __int64 *)v190 )
            {
              v144 = s1.m128i_i32[0];
              _libc_free(v188, a2);
              v124 = v144;
              v125 = v154;
            }
            LODWORD(v189) = v189 + 1;
            v188 = v125;
            HIDWORD(v189) = v124;
          }
          else
          {
LABEL_162:
            v75 = &v188[3 * v79];
            if ( v75 )
            {
              v80 = v171;
              *v75 = 6;
              v75[1] = 0;
              v75[2] = (unsigned __int64)v80;
              if ( v80 != 0 && v80 + 4096 != 0 && v80 != (unsigned __int8 *)-8192LL )
LABEL_165:
                sub_BD73F0((__int64)v75);
LABEL_144:
              v74 = v189;
            }
LABEL_145:
            LODWORD(v189) = v74 + 1;
          }
LABEL_88:
          v12 = *(_QWORD *)(v12 + 8);
          if ( v12 )
            continue;
          v9 = v176;
          v8 = v175;
LABEL_90:
          if ( !v9 )
          {
            if ( v8 != v177 )
              _libc_free(v8, a2);
            if ( (_DWORD)v189 )
            {
              v44 = 0;
              v45 = 24LL * (unsigned int)v189;
              do
              {
                while ( 1 )
                {
                  v46 = (__int64 *)v188[v44 / 8 + 2];
                  if ( v46 )
                  {
                    if ( *(_BYTE *)v46 == 85 )
                    {
                      v47 = *(v46 - 4);
                      if ( v47 )
                      {
                        if ( !*(_BYTE *)v47
                          && *(_QWORD *)(v47 + 24) == v46[10]
                          && (*(_BYTE *)(v47 + 33) & 0x20) != 0
                          && *(_DWORD *)(v47 + 36) == 282 )
                        {
                          v48 = a1[7];
                          v49 = a1[9];
                          v50 = v188[v44 / 8 + 2];
                          v51 = a1[11];
                          s1.m128i_i64[0] = (__int64)v179;
                          s1.m128i_i64[1] = 0x600000000LL;
                          v166 = sub_D640E0(v50, v51, v49, v48, 1, (__int64)&s1);
                          if ( s1.m128i_i64[0] != s1.m128i_i64[0] + 8LL * s1.m128i_u32[2] )
                          {
                            v161 = v46;
                            v52 = s1.m128i_i64[0] + 8LL * s1.m128i_u32[2];
                            v53 = (__int64 *)s1.m128i_i64[0];
                            do
                            {
                              v54 = *v53++;
                              v55 = a1[5] + 2096LL;
                              v175 = (_BYTE *)v54;
                              sub_F200C0(v55, (__int64 *)&v175);
                            }
                            while ( (__int64 *)v52 != v53 );
                            v46 = v161;
                          }
                          sub_F162A0((__int64)a1, (__int64)v46, v166);
                          v56 = v46;
                          sub_F207A0((__int64)a1, v46);
                          v57 = &v188[v44 / 8];
                          v58 = v188[v44 / 8 + 2];
                          if ( v58 )
                          {
                            if ( v58 != -8192 && v58 != -4096 )
                              sub_BD60C0(&v188[v44 / 8]);
                            v57[2] = 0;
                          }
                          if ( (_BYTE *)s1.m128i_i64[0] != v179 )
                            break;
                        }
                      }
                    }
                  }
                  v44 += 24LL;
                  if ( v45 == v44 )
                    goto LABEL_114;
                }
                _libc_free(s1.m128i_i64[0], v56);
                v44 += 24LL;
              }
              while ( v45 != v44 );
LABEL_114:
              if ( (_DWORD)v189 )
              {
                v59 = 0;
                v167 = 24LL * (unsigned int)v189;
                do
                {
                  v64 = *(unsigned __int64 *)((char *)v188 + v59 + 16);
                  if ( v64 )
                  {
                    if ( *(_BYTE *)v64 == 82 )
                    {
                      v60 = sub_B53600(*(_WORD *)(v64 + 2) & 0x3F);
                      v61 = (_QWORD *)sub_BD5C60(v64);
                      v62 = sub_BCB2A0(v61);
                      v63 = sub_ACD640(v62, v60, 0);
                      sub_F162A0((__int64)a1, v64, v63);
                    }
                    else if ( *(_BYTE *)v64 == 62 )
                    {
                      v65 = v182;
                      if ( &v182[(unsigned int)v183] != v182 )
                      {
                        v162 = v59;
                        v66 = *(unsigned __int64 *)((char *)v188 + v59 + 16);
                        v67 = &v182[(unsigned int)v183];
                        do
                        {
                          v68 = *v65;
                          v69 = *(_QWORD *)(*v65 - 32);
                          if ( !v69 || *(_BYTE *)v69 || *(_QWORD *)(v69 + 24) != *(_QWORD *)(v68 + 80) )
                            BUG();
                          if ( *(_DWORD *)(v69 + 36) == 69 )
                            sub_F519F0(v68, v66, v169);
                          ++v65;
                        }
                        while ( v67 != v65 );
                        v64 = v66;
                        v59 = v162;
                      }
                      v70 = v185;
                      if ( &v185[8 * (unsigned int)v186] != v185 )
                      {
                        v163 = v59;
                        v71 = &v185[8 * (unsigned int)v186];
                        do
                        {
                          while ( 1 )
                          {
                            v72 = *(_QWORD *)v70;
                            if ( !*(_BYTE *)(*(_QWORD *)v70 + 64LL) )
                              break;
                            v70 += 8;
                            if ( v71 == v70 )
                              goto LABEL_136;
                          }
                          v70 += 8;
                          sub_F51C80(v72, v64, v169);
                        }
                        while ( v71 != v70 );
LABEL_136:
                        v59 = v163;
                      }
                    }
                    else
                    {
                      v115 = sub_ACADE0(*(__int64 ***)(v64 + 8));
                      sub_F162A0((__int64)a1, v64, v115);
                    }
                    sub_F207A0((__int64)a1, (__int64 *)v64);
                  }
                  v59 += 24;
                }
                while ( v167 != v59 );
              }
            }
            if ( *v165 == 34 )
            {
              v101 = 0;
              v102 = (__int64 *)sub_B43CA0((__int64)v165);
              v103 = sub_B6E160(v102, 0x49u, 0, 0);
              sub_B43C20((__int64)&v175, *((_QWORD *)v165 + 5));
              v180 = 257;
              v104 = *((_QWORD *)v165 - 8);
              v105 = *((_QWORD *)v165 - 12);
              if ( v103 )
                v101 = *(_QWORD *)(v103 + 24);
              v164 = (__int64)v175;
              v168 = v176;
              v106 = sub_BD2CC0(88, 3u);
              if ( v106 )
              {
                sub_B44260((__int64)v106, **(_QWORD **)(v101 + 16), 5, 3u, v164, v168);
                v106[9] = 0;
                sub_B4A9C0((__int64)v106, v101, v103, v105, v104, (__int64)&s1, 0, 0, 0, 0);
              }
            }
            v107 = v182;
            v108 = &v182[(unsigned int)v183];
            if ( v108 != v182 )
            {
              do
              {
                v109 = *v107;
                v110 = *(_QWORD *)(*v107 - 32);
                if ( !v110 || *(_BYTE *)v110 || *(_QWORD *)(v110 + 24) != *(_QWORD *)(v109 + 80) )
                  BUG();
                if ( *(_DWORD *)(v110 + 36) == 69
                  || sub_AF4730(*(_QWORD *)(*(_QWORD *)(v109 + 32 * (2LL - (*(_DWORD *)(v109 + 4) & 0x7FFFFFF))) + 24LL)) )
                {
                  sub_B43D60((_QWORD *)v109);
                }
                ++v107;
              }
              while ( v108 != v107 );
            }
            v111 = v185;
            v112 = &v185[8 * (unsigned int)v186];
            if ( v112 != v185 )
            {
              do
              {
                v113 = *(_QWORD **)v111;
                if ( !*(_BYTE *)(*(_QWORD *)v111 + 64LL) || (v114 = sub_B11F60((__int64)(v113 + 10)), sub_AF4730(v114)) )
                  sub_B14290(v113);
                v111 += 8;
              }
              while ( v112 != v111 );
            }
            a2 = (__int64)v165;
            v14 = sub_F207A0((__int64)a1, (__int64 *)v165);
            goto LABEL_11;
          }
          goto LABEL_5;
        }
        v148 = s1.m128i_i32[2];
        if ( v148 == (unsigned int)sub_C444A0((__int64)&s1) )
        {
          if ( s1.m128i_i64[0] )
            j_j___libc_free_0_0(s1.m128i_i64[0]);
          goto LABEL_161;
        }
        if ( s1.m128i_i64[0] )
          j_j___libc_free_0_0(s1.m128i_i64[0]);
LABEL_8:
        if ( v175 != v177 )
          _libc_free(v175, a2);
        v14 = 0;
LABEL_11:
        if ( v169 )
        {
          v15 = *(unsigned int *)(v169 + 424);
          if ( (_DWORD)v15 )
          {
            v16 = *(_QWORD *)(v169 + 408);
            v17 = v16 + 56 * v15;
            do
            {
              if ( *(_QWORD *)v16 != -8192 && *(_QWORD *)v16 != -4096 )
              {
                v18 = *(_QWORD *)(v16 + 8);
                v19 = v18 + 8LL * *(unsigned int *)(v16 + 16);
                if ( v18 != v19 )
                {
                  do
                  {
                    a2 = *(_QWORD *)(v19 - 8);
                    v19 -= 8;
                    if ( a2 )
                      sub_B91220(v19, a2);
                  }
                  while ( v18 != v19 );
                  v19 = *(_QWORD *)(v16 + 8);
                }
                if ( v19 != v16 + 24 )
                  _libc_free(v19, a2);
              }
              v16 += 56;
            }
            while ( v17 != v16 );
            v15 = *(unsigned int *)(v169 + 424);
          }
          v20 = 56 * v15;
          sub_C7D6A0(*(_QWORD *)(v169 + 408), 56 * v15, 8);
          v21 = *(_QWORD *)(v169 + 344);
          v22 = v21 + 8LL * *(unsigned int *)(v169 + 352);
          if ( v21 != v22 )
          {
            do
            {
              v20 = *(_QWORD *)(v22 - 8);
              v22 -= 8;
              if ( v20 )
                sub_B91220(v22, v20);
            }
            while ( v21 != v22 );
            v22 = *(_QWORD *)(v169 + 344);
          }
          if ( v22 != v169 + 360 )
            _libc_free(v22, v20);
          v23 = *(_QWORD *)(v169 + 328);
          v24 = v23 + 56LL * *(unsigned int *)(v169 + 336);
          if ( v23 != v24 )
          {
            do
            {
              v24 -= 56;
              v25 = *(_QWORD *)(v24 + 40);
              if ( v25 != v24 + 56 )
                _libc_free(v25, v20);
              v20 = 8LL * *(unsigned int *)(v24 + 32);
              sub_C7D6A0(*(_QWORD *)(v24 + 16), v20, 8);
            }
            while ( v23 != v24 );
            v24 = *(_QWORD *)(v169 + 328);
          }
          if ( v24 != v169 + 344 )
            _libc_free(v24, v20);
          v26 = 16LL * *(unsigned int *)(v169 + 320);
          sub_C7D6A0(*(_QWORD *)(v169 + 304), v26, 8);
          v27 = *(_QWORD *)(v169 + 248);
          v28 = v27 + 8LL * *(unsigned int *)(v169 + 256);
          if ( v27 != v28 )
          {
            do
            {
              v26 = *(_QWORD *)(v28 - 8);
              v28 -= 8;
              if ( v26 )
                sub_B91220(v28, v26);
            }
            while ( v27 != v28 );
            v28 = *(_QWORD *)(v169 + 248);
          }
          if ( v28 != v169 + 264 )
            _libc_free(v28, v26);
          v29 = *(_QWORD *)(v169 + 200);
          if ( v29 != v169 + 216 )
            _libc_free(v29, v26);
          v30 = *(_QWORD *)(v169 + 152);
          if ( v30 != v169 + 168 )
            _libc_free(v30, v26);
          v31 = *(_QWORD *)(v169 + 104);
          v32 = v31 + 8LL * *(unsigned int *)(v169 + 112);
          if ( v31 != v32 )
          {
            do
            {
              v26 = *(_QWORD *)(v32 - 8);
              v32 -= 8;
              if ( v26 )
                sub_B91220(v32, v26);
            }
            while ( v31 != v32 );
            v32 = *(_QWORD *)(v169 + 104);
          }
          if ( v32 != v169 + 120 )
            _libc_free(v32, v26);
          v33 = *(_QWORD *)(v169 + 56);
          v34 = v33 + 8LL * *(unsigned int *)(v169 + 64);
          if ( v33 != v34 )
          {
            do
            {
              v26 = *(_QWORD *)(v34 - 8);
              v34 -= 8;
              if ( v26 )
                sub_B91220(v34, v26);
            }
            while ( v33 != v34 );
            v34 = *(_QWORD *)(v169 + 56);
          }
          if ( v34 != v169 + 72 )
            _libc_free(v34, v26);
          a2 = 432;
          j_j___libc_free_0(v169, 432);
        }
        if ( v185 != v187 )
          _libc_free(v185, a2);
        if ( v182 != (__int64 *)v184 )
          _libc_free(v182, a2);
        v35 = v188;
        v36 = &v188[3 * (unsigned int)v189];
        if ( v188 != v36 )
        {
          do
          {
            v37 = *(v36 - 1);
            v36 -= 3;
            if ( v37 != 0 && v37 != -4096 && v37 != -8192 )
              sub_BD60C0(v36);
          }
          while ( v35 != v36 );
          v36 = v188;
        }
        if ( v36 != (unsigned __int64 *)v190 )
          _libc_free(v36, a2);
        return v14;
      case 'U':
        v83 = *((_QWORD *)v13 - 4);
        if ( !v83
          || *(_BYTE *)v83
          || *(_QWORD *)(v83 + 24) != *((_QWORD *)v13 + 10)
          || (*(_BYTE *)(v83 + 33) & 0x20) == 0 )
        {
          if ( !*((_QWORD *)v13 + 2) )
          {
            v155 = (__int64)v13;
            v126 = sub_B46900(v13);
            v13 = (unsigned __int8 *)v155;
            if ( v126 )
            {
              v127 = sub_A73ED0((_QWORD *)(v155 + 72), 41);
              v128 = (unsigned __int8 *)v155;
              if ( v127 || (v129 = sub_B49560(v155, 41), v128 = (unsigned __int8 *)v155, v129) )
              {
                a2 = (__int64)v128;
                sub_D67230(&s1, v128, v3);
                if ( v181 )
                {
                  if ( v11 == s1.m128i_i64[0] )
                  {
                    v79 = (unsigned int)v189;
                    v74 = v189;
                    if ( HIDWORD(v189) > (unsigned int)v189 )
                      goto LABEL_162;
LABEL_262:
                    v156 = (unsigned __int64 *)sub_C8D7D0(
                                                 (__int64)&v188,
                                                 (__int64)v190,
                                                 0,
                                                 0x18u,
                                                 (unsigned __int64 *)&s1,
                                                 (__int64)&v188);
                    v134 = &v156[3 * (unsigned int)v189];
                    if ( v134 )
                    {
                      v135 = v171;
                      *v134 = 6;
                      v134[1] = 0;
                      v134[2] = (unsigned __int64)v135;
                      if ( v135 != 0 && v135 + 4096 != 0 && v135 != (unsigned __int8 *)-8192LL )
                        sub_BD73F0((__int64)v134);
                    }
                    a2 = (__int64)v156;
                    sub_F17F80((__int64)&v188, v156);
                    v136 = s1.m128i_i32[0];
                    if ( v188 != (unsigned __int64 *)v190 )
                    {
                      v145 = s1.m128i_i32[0];
                      _libc_free(v188, v156);
                      v136 = v145;
                    }
                    LODWORD(v189) = v189 + 1;
                    HIDWORD(v189) = v136;
                    v188 = v156;
                    goto LABEL_88;
                  }
                }
              }
              v13 = v171;
            }
          }
          a2 = (__int64)v3;
          if ( v11 == sub_D5D560((__int64)v13, v3) )
          {
            a2 = (__int64)v171;
            sub_D5D330((__int64)&s1, (__int64)v171, v3);
            if ( v179[0] == v160 )
            {
              if ( !v160
                || (v130 = s1.m128i_i64[1], v173 == s1.m128i_i64[1])
                && (!s1.m128i_i64[1] || (a2 = (__int64)s2, !memcmp((const void *)s1.m128i_i64[0], s2, s1.m128i_u64[1]))) )
              {
                a2 = (__int64)&v171;
                sub_F18070((__int64)&v188, (unsigned __int64 *)&v171, v130, v131, v132, v133);
                goto LABEL_88;
              }
            }
          }
          if ( v11 != sub_D5CCF0(v171) )
            goto LABEL_8;
          a2 = (__int64)v171;
          sub_D5D330((__int64)&s1, (__int64)v171, v3);
          if ( v179[0] != v160 )
            goto LABEL_8;
          if ( v160 )
          {
            v85 = s1.m128i_i64[1];
            if ( v173 != s1.m128i_i64[1] )
              goto LABEL_8;
            if ( s1.m128i_i64[1] )
            {
              a2 = (__int64)s2;
              if ( memcmp((const void *)s1.m128i_i64[0], s2, s1.m128i_u64[1]) )
                goto LABEL_8;
            }
          }
          a2 = (__int64)&v171;
          sub_F18070((__int64)&v188, (unsigned __int64 *)&v171, v85, v86, v87, v88);
          v13 = v171;
          goto LABEL_85;
        }
        v84 = *(_DWORD *)(v83 + 36);
        if ( v84 > 0xF3 )
        {
          if ( v84 != 282 )
          {
            if ( v84 != 346 )
              goto LABEL_8;
LABEL_205:
            v39 = (unsigned int)v189;
            v40 = v189;
            if ( HIDWORD(v189) > (unsigned int)v189 )
            {
LABEL_79:
              v41 = &v188[3 * v39];
              if ( v41 )
              {
                *v41 = 6;
                v41[1] = 0;
                v41[2] = (unsigned __int64)v13;
                if ( v13 != (unsigned __int8 *)-8192LL && v13 != (unsigned __int8 *)-4096LL )
                  sub_BD73F0((__int64)v41);
                v40 = v189;
                v13 = v171;
              }
              LODWORD(v189) = v40 + 1;
            }
            else
            {
              v150 = (unsigned __int64 *)sub_C8D7D0(
                                           (__int64)&v188,
                                           (__int64)v190,
                                           0,
                                           0x18u,
                                           (unsigned __int64 *)&s1,
                                           (__int64)&v188);
              v98 = &v150[3 * (unsigned int)v189];
              if ( v98 )
              {
                v99 = v171;
                v98[1] = 0;
                *v98 = 6;
                v98[2] = (unsigned __int64)v99;
                if ( v99 != 0 && v99 + 4096 != 0 && v99 != (unsigned __int8 *)-8192LL )
                  sub_BD73F0((__int64)v98);
              }
              a2 = (__int64)v150;
              sub_F17F80((__int64)&v188, v150);
              v100 = s1.m128i_i32[0];
              if ( v188 != (unsigned __int64 *)v190 )
              {
                v142 = s1.m128i_i32[0];
                _libc_free(v188, v150);
                v100 = v142;
              }
              LODWORD(v189) = v189 + 1;
              HIDWORD(v189) = v100;
              v13 = v171;
              v188 = v150;
            }
LABEL_85:
            v42 = (unsigned int)v176;
            v43 = (unsigned int)v176 + 1LL;
            if ( v43 > HIDWORD(v176) )
            {
              a2 = (__int64)v177;
              v147 = v13;
              sub_C8D5F0((__int64)&v175, v177, v43, 8u, (__int64)v13, v5);
              v42 = (unsigned int)v176;
              v13 = v147;
            }
            *(_QWORD *)&v175[8 * v42] = v13;
            LODWORD(v176) = v176 + 1;
            goto LABEL_88;
          }
        }
        else
        {
          if ( v84 > 0xCB )
          {
            switch ( v84 )
            {
              case 0xCCu:
              case 0xCDu:
              case 0xD2u:
              case 0xD3u:
                goto LABEL_140;
              case 0xD0u:
                goto LABEL_205;
              case 0xEEu:
              case 0xF1u:
              case 0xF3u:
                v94 = *((_DWORD *)v13 + 1) & 0x7FFFFFF;
                v95 = *(_QWORD *)&v13[32 * (3 - v94)];
                if ( *(_DWORD *)(v95 + 32) <= 0x40u )
                {
                  if ( *(_QWORD *)(v95 + 24) )
                    goto LABEL_8;
                }
                else
                {
                  v140 = v13;
                  v141 = *(_DWORD *)(v95 + 32);
                  v149 = *((_DWORD *)v13 + 1) & 0x7FFFFFF;
                  v96 = sub_C444A0(v95 + 24);
                  v94 = v149;
                  v13 = v140;
                  if ( v141 != v96 )
                    goto LABEL_8;
                }
                v97 = *(_QWORD *)&v13[-32 * v94];
                if ( v11 == v97 && v97 )
                  goto LABEL_140;
                goto LABEL_8;
              default:
                goto LABEL_8;
            }
          }
          if ( v84 != 11 )
            goto LABEL_8;
        }
LABEL_140:
        v74 = v189;
        if ( HIDWORD(v189) <= (unsigned int)v189 )
          goto LABEL_262;
        v75 = &v188[3 * (unsigned int)v189];
        if ( !v75 )
          goto LABEL_145;
        *v75 = 6;
        v75[1] = 0;
        v75[2] = (unsigned __int64)v13;
        if ( v13 == (unsigned __int8 *)-8192LL || v13 == (unsigned __int8 *)-4096LL )
          goto LABEL_144;
        goto LABEL_165;
      default:
        goto LABEL_8;
    }
  }
}

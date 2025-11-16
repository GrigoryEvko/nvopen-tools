// Function: sub_34F3C10
// Address: 0x34f3c10
//
__int64 __fastcall sub_34F3C10(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  int v5; // eax
  __int64 v6; // rbx
  _BYTE *v7; // r14
  _BYTE *v8; // r13
  bool v9; // al
  __int64 v10; // rbx
  __int64 v11; // r9
  __int64 v12; // r13
  __int64 v13; // r8
  int v14; // eax
  __int64 v15; // rdi
  _QWORD *v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int32 v20; // eax
  __int64 v21; // rsi
  __int32 v22; // r12d
  __int64 v23; // rcx
  __int64 v24; // rcx
  char v25; // al
  _BYTE *v27; // r12
  unsigned int v28; // eax
  __int64 v29; // r12
  __int64 v30; // rax
  __int64 v31; // r8
  __int64 v32; // r9
  unsigned int v33; // eax
  __int64 v34; // r8
  __int64 v35; // r9
  bool v36; // zf
  unsigned int v37; // ebx
  unsigned int *v38; // rcx
  unsigned __int64 v39; // rdx
  unsigned int *v40; // r13
  unsigned int *v41; // rax
  unsigned __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // r14
  int v45; // r11d
  _DWORD *v46; // rax
  _DWORD *v47; // rdx
  __int64 *v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rdi
  __int64 v55; // rsi
  __int64 v56; // r8
  __int64 v57; // r9
  unsigned int *v58; // rbx
  int v59; // r11d
  __int64 v60; // rax
  unsigned int *v61; // rsi
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rdi
  __int64 (*v65)(); // rax
  unsigned __int64 v66; // rdi
  __int64 v67; // rax
  __int64 v68; // rsi
  __int64 v69; // rcx
  __int64 v70; // rdx
  __int64 v71; // rax
  __int64 v72; // rdi
  unsigned int *v73; // rcx
  __int64 (*v74)(); // rax
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  __int32 v78; // eax
  unsigned __int8 *v79; // rsi
  __int64 v80; // rax
  __int64 v81; // rcx
  _QWORD *v82; // rsi
  __int64 v83; // rdx
  __int64 v84; // rdi
  __int64 v85; // rsi
  __int64 (*v86)(); // rax
  __int32 v87; // eax
  unsigned __int8 *v88; // rsi
  __int64 v89; // rcx
  __int64 v90; // rax
  __int64 v91; // rdi
  __int64 (*v92)(); // rax
  __int64 v93; // rax
  __int64 v94; // rdi
  __int64 (*v95)(); // rax
  __int64 v96; // rax
  __int64 v97; // rdi
  __int64 (*v98)(); // rax
  __int64 v99; // rax
  __int64 v100; // rsi
  __int64 v101; // r13
  __int64 v102; // r13
  unsigned int *v103; // r15
  _QWORD *v104; // rbx
  bool v105; // r9
  __int64 v106; // rax
  __int64 v107; // rax
  __int64 v108; // rdx
  _QWORD *v109; // r12
  __int64 v110; // r9
  __int64 v111; // rdi
  __int64 (*v112)(); // rax
  __int64 v113; // rax
  __int64 v114; // rdi
  __int64 (*v115)(); // rax
  __int64 v116; // rax
  unsigned int *v120; // [rsp+38h] [rbp-148h]
  __int64 v121; // [rsp+48h] [rbp-138h]
  __int64 v122; // [rsp+48h] [rbp-138h]
  unsigned int v123; // [rsp+50h] [rbp-130h]
  __int32 v124; // [rsp+58h] [rbp-128h]
  char v125; // [rsp+6Eh] [rbp-112h]
  unsigned __int8 v126; // [rsp+6Fh] [rbp-111h]
  int v127; // [rsp+70h] [rbp-110h]
  __int64 v128; // [rsp+70h] [rbp-110h]
  __int32 v129; // [rsp+70h] [rbp-110h]
  char v130; // [rsp+78h] [rbp-108h]
  __int32 v131; // [rsp+78h] [rbp-108h]
  unsigned __int64 v132; // [rsp+80h] [rbp-100h]
  __int64 v133; // [rsp+88h] [rbp-F8h]
  int v134; // [rsp+90h] [rbp-F0h]
  unsigned int *v135; // [rsp+90h] [rbp-F0h]
  unsigned int *v136; // [rsp+90h] [rbp-F0h]
  int v137; // [rsp+90h] [rbp-F0h]
  char v138; // [rsp+90h] [rbp-F0h]
  int v139; // [rsp+90h] [rbp-F0h]
  int v140; // [rsp+90h] [rbp-F0h]
  __int64 v141; // [rsp+98h] [rbp-E8h]
  _BYTE *v142; // [rsp+98h] [rbp-E8h]
  __int64 v143; // [rsp+98h] [rbp-E8h]
  bool v144; // [rsp+A0h] [rbp-E0h]
  __int64 v145; // [rsp+A0h] [rbp-E0h]
  unsigned __int8 *v147; // [rsp+B8h] [rbp-C8h] BYREF
  unsigned __int8 *v148[4]; // [rsp+C0h] [rbp-C0h] BYREF
  __m128i v149; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v150; // [rsp+F0h] [rbp-90h]
  __int64 v151; // [rsp+F8h] [rbp-88h]
  __int64 v152; // [rsp+100h] [rbp-80h]
  unsigned int *v153; // [rsp+110h] [rbp-70h] BYREF
  __int64 v154; // [rsp+118h] [rbp-68h]
  _QWORD v155[12]; // [rsp+120h] [rbp-60h] BYREF

  v133 = a3 + 48;
  v126 = 0;
  if ( *(_QWORD *)(a3 + 56) != a3 + 48 )
  {
    v4 = *(_QWORD *)(a3 + 56);
    while ( 1 )
    {
      v5 = sub_2E88FE0(v4);
      v6 = *(_QWORD *)(v4 + 32);
      if ( v5 + *(unsigned __int8 *)(*(_QWORD *)(v4 + 16) + 9LL) )
      {
        if ( !*(_BYTE *)v6 && (*(_BYTE *)(v6 + 3) & 0x10) != 0 && (*(_WORD *)(v6 + 2) & 0xFF0) != 0 )
        {
          v28 = sub_2E89F40(v4, 0);
          v6 = *(_QWORD *)(v4 + 32);
          v29 = v6 + 40LL * v28;
          if ( !*(_DWORD *)(v29 + 8) )
            break;
        }
      }
LABEL_4:
      v7 = (_BYTE *)(v6 + 40LL * (*(_DWORD *)(v4 + 40) & 0xFFFFFF));
      if ( v7 == (_BYTE *)v6 )
        goto LABEL_37;
      v8 = (_BYTE *)v6;
      while ( 1 )
      {
        v9 = sub_2DADC00(v8);
        if ( v9 )
          break;
        v8 += 40;
        if ( v7 == v8 )
          goto LABEL_37;
      }
      v144 = v9;
      if ( v7 != v8 )
      {
        while ( *v8 || (v8[4] & 4) == 0 )
        {
          v27 = v8 + 40;
          if ( v7 != v8 + 40 )
          {
            while ( 1 )
            {
              v8 = v27;
              if ( sub_2DADC00(v27) )
                break;
              v27 += 40;
              if ( v7 == v27 )
                goto LABEL_37;
            }
            if ( v27 != v7 )
              continue;
          }
          goto LABEL_37;
        }
        v125 = *(_BYTE *)(*(_QWORD *)(a1 + 208) + 48LL);
        if ( v125 )
        {
          v43 = 5LL * (unsigned int)sub_2E88FE0(v4);
          if ( v7 == (_BYTE *)(v6 + 8 * v43) )
            goto LABEL_79;
          v130 = 0;
          v142 = v7;
          v44 = v6 + 8 * v43;
LABEL_66:
          while ( 2 )
          {
            while ( 2 )
            {
              if ( *(_BYTE *)v44 )
                goto LABEL_77;
              v45 = *(_DWORD *)(v44 + 8);
              if ( v45 >= 0 || (*(_WORD *)(v44 + 2) & 0xFF0) != 0 )
                goto LABEL_77;
              if ( *(_QWORD *)(a1 + 320) )
              {
                v67 = *(_QWORD *)(a1 + 296);
                if ( v67 )
                {
                  v68 = a1 + 288;
                  do
                  {
                    while ( 1 )
                    {
                      v69 = *(_QWORD *)(v67 + 16);
                      v70 = *(_QWORD *)(v67 + 24);
                      if ( (unsigned int)v45 <= *(_DWORD *)(v67 + 32) )
                        break;
                      v67 = *(_QWORD *)(v67 + 24);
                      if ( !v70 )
                        goto LABEL_92;
                    }
                    v68 = v67;
                    v67 = *(_QWORD *)(v67 + 16);
                  }
                  while ( v69 );
LABEL_92:
                  if ( v68 != a1 + 288 && (unsigned int)v45 >= *(_DWORD *)(v68 + 32) )
                  {
                    v44 += 40;
                    if ( v142 != (_BYTE *)v44 )
                      continue;
                    goto LABEL_78;
                  }
                }
              }
              else
              {
                v46 = *(_DWORD **)(a1 + 232);
                v47 = &v46[*(unsigned int *)(a1 + 240)];
                if ( v46 != v47 )
                {
                  while ( v45 != *v46 )
                  {
                    if ( v47 == ++v46 )
                      goto LABEL_75;
                  }
                  if ( v47 != v46 )
                    goto LABEL_77;
                }
              }
              break;
            }
LABEL_75:
            v48 = (__int64 *)(*(_QWORD *)(a4 + 16) + 32LL * (v45 & 0x7FFFFFFF));
            v49 = *v48;
            v50 = v48[2];
            v51 = v48[1];
            v52 = v48[3];
            if ( v49 == v50 && v51 == v52 )
              goto LABEL_77;
            v53 = v49 & ~v50;
            v54 = *(_QWORD *)(a1 + 224);
            v55 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 208) + 56LL) + 16LL * (v45 & 0x7FFFFFFF));
            v153 = (unsigned int *)v155;
            v134 = v45;
            v154 = 0xC00000000LL;
            v132 = v55 & 0xFFFFFFFFFFFFFFF8LL;
            sub_2FF7100(v54, v55 & 0xFFFFFFFFFFFFFFF8LL, v53, ~v52 & v51, (__int64)&v153);
            v58 = v153;
            v59 = v134;
            v60 = 4LL * (unsigned int)v154;
            v61 = &v153[(unsigned __int64)v60 / 4];
            v62 = v60 >> 2;
            v63 = v60 >> 4;
            if ( !v63 )
              goto LABEL_126;
            v135 = &v153[4 * v63];
            while ( 1 )
            {
              v64 = *(_QWORD *)(a1 + 224);
              v65 = *(__int64 (**)())(*(_QWORD *)v64 + 280LL);
              if ( v65 == sub_2FF5260 )
                goto LABEL_83;
              v127 = v59;
              v71 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD))v65)(v64, v132, *v58);
              v59 = v127;
              if ( !v71 )
                goto LABEL_83;
              v72 = *(_QWORD *)(a1 + 224);
              v73 = v58 + 1;
              v74 = *(__int64 (**)())(*(_QWORD *)v72 + 280LL);
              if ( v74 == sub_2FF5260 )
                break;
              v90 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD))v74)(v72, v132, v58[1]);
              v59 = v127;
              v73 = v58 + 1;
              if ( !v90 )
                break;
              v91 = *(_QWORD *)(a1 + 224);
              v73 = v58 + 2;
              v92 = *(__int64 (**)())(*(_QWORD *)v91 + 280LL);
              if ( v92 == sub_2FF5260 )
                break;
              v93 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD))v92)(v91, v132, v58[2]);
              v59 = v127;
              v73 = v58 + 2;
              if ( !v93 )
                break;
              v94 = *(_QWORD *)(a1 + 224);
              v73 = v58 + 3;
              v95 = *(__int64 (**)())(*(_QWORD *)v94 + 280LL);
              if ( v95 == sub_2FF5260 )
                break;
              v96 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD))v95)(v94, v132, v58[3]);
              v59 = v127;
              v73 = v58 + 3;
              if ( !v96 )
                break;
              v58 += 4;
              if ( v135 == v58 )
              {
                v62 = v61 - v58;
LABEL_126:
                switch ( v62 )
                {
                  case 2LL:
                    goto LABEL_162;
                  case 3LL:
                    v111 = *(_QWORD *)(a1 + 224);
                    v112 = *(__int64 (**)())(*(_QWORD *)v111 + 280LL);
                    if ( v112 != sub_2FF5260 )
                    {
                      v139 = v59;
                      v113 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD))v112)(v111, v132, *v58);
                      v59 = v139;
                      if ( v113 )
                      {
                        ++v58;
LABEL_162:
                        v114 = *(_QWORD *)(a1 + 224);
                        v115 = *(__int64 (**)())(*(_QWORD *)v114 + 280LL);
                        if ( v115 != sub_2FF5260 )
                        {
                          v140 = v59;
                          v116 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD))v115)(v114, v132, *v58);
                          v59 = v140;
                          if ( v116 )
                          {
                            ++v58;
                            goto LABEL_129;
                          }
                        }
                      }
                    }
LABEL_83:
                    if ( v61 == v58 )
                      break;
                    goto LABEL_84;
                  case 1LL:
LABEL_129:
                    v97 = *(_QWORD *)(a1 + 224);
                    v98 = *(__int64 (**)())(*(_QWORD *)v97 + 280LL);
                    if ( v98 == sub_2FF5260 )
                      goto LABEL_83;
                    v137 = v59;
                    v99 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD))v98)(v97, v132, *v58);
                    v59 = v137;
                    if ( !v99 )
                      goto LABEL_83;
                    break;
                }
LABEL_99:
                v120 = &v153[(unsigned int)v154];
                if ( v153 != v120 )
                {
                  v136 = v153;
                  v124 = v59;
                  do
                  {
                    v84 = *(_QWORD *)(a1 + 224);
                    v85 = 0;
                    v123 = *v136;
                    v86 = *(__int64 (**)())(*(_QWORD *)v84 + 280LL);
                    if ( v86 != sub_2FF5260 )
                      v85 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD))v86)(v84, v132, v123);
                    v87 = sub_2EC06C0(*(_QWORD *)(a1 + 208), v85, byte_3F871B3, 0, v56, v57);
                    v88 = *(unsigned __int8 **)(v4 + 56);
                    v131 = v87;
                    v89 = *(_QWORD *)(*(_QWORD *)(a1 + 200) + 8LL);
                    v148[0] = v88;
                    v75 = v89 - 440;
                    if ( v88 )
                    {
                      v128 = v75;
                      sub_B96E90((__int64)v148, (__int64)v88, 1);
                      v75 = v128;
                      v149.m128i_i64[0] = (__int64)v148[0];
                      if ( v148[0] )
                      {
                        sub_B976B0((__int64)v148, v148[0], (__int64)&v149);
                        v148[0] = 0;
                        v75 = v128;
                      }
                    }
                    else
                    {
                      v149.m128i_i64[0] = 0;
                    }
                    v149.m128i_i64[1] = 0;
                    v150 = 0;
                    sub_2F2A600(*(_QWORD *)(v4 + 24), v4, v149.m128i_i64, v75, v131);
                    if ( v149.m128i_i64[0] )
                      sub_B91220((__int64)&v149, v149.m128i_i64[0]);
                    if ( v148[0] )
                      sub_B91220((__int64)v148, (__int64)v148[0]);
                    v78 = sub_2EC06C0(*(_QWORD *)(a1 + 208), v132, byte_3F871B3, 0, v76, v77);
                    v79 = *(unsigned __int8 **)(v4 + 56);
                    v129 = v78;
                    v80 = *(_QWORD *)(a1 + 200);
                    v147 = v79;
                    v81 = *(_QWORD *)(v80 + 8) - 360LL;
                    if ( v79 )
                    {
                      v121 = *(_QWORD *)(v80 + 8) - 360LL;
                      sub_B96E90((__int64)&v147, (__int64)v79, 1);
                      v81 = v121;
                      v148[0] = v147;
                      if ( v147 )
                      {
                        sub_B976B0((__int64)&v147, v147, (__int64)v148);
                        v147 = 0;
                        v81 = v121;
                      }
                    }
                    else
                    {
                      v148[0] = 0;
                    }
                    v148[1] = 0;
                    v148[2] = 0;
                    v82 = sub_2F2A600(*(_QWORD *)(v4 + 24), v4, (__int64 *)v148, v81, v129);
                    v149.m128i_i32[2] = v124;
                    v122 = v83;
                    v150 = 0;
                    v151 = 0;
                    v152 = 0;
                    v149.m128i_i64[0] = 0;
                    sub_2E8EAD0(v83, (__int64)v82, &v149);
                    v152 = 0;
                    v149.m128i_i32[2] = v131;
                    v150 = 0;
                    v151 = 0;
                    v149.m128i_i64[0] = 0;
                    sub_2E8EAD0(v122, (__int64)v82, &v149);
                    v150 = 0;
                    v149.m128i_i64[0] = 1;
                    v151 = v123;
                    sub_2E8EAD0(v122, (__int64)v82, &v149);
                    if ( v148[0] )
                      sub_B91220((__int64)v148, (__int64)v148[0]);
                    if ( v147 )
                      sub_B91220((__int64)&v147, (__int64)v147);
                    ++v136;
                    v124 = v129;
                  }
                  while ( v120 != v136 );
                  v59 = v129;
                  v130 = v125;
                }
                sub_2EAB0C0(v44, v59);
                v66 = (unsigned __int64)v153;
                if ( v153 != (unsigned int *)v155 )
                  goto LABEL_85;
LABEL_77:
                v44 += 40;
                if ( v142 == (_BYTE *)v44 )
                  goto LABEL_78;
                goto LABEL_66;
              }
            }
            if ( v61 == v73 )
              goto LABEL_99;
LABEL_84:
            v66 = (unsigned __int64)v153;
            if ( v153 == (unsigned int *)v155 )
              goto LABEL_77;
LABEL_85:
            _libc_free(v66);
            v44 += 40;
            if ( v142 != (_BYTE *)v44 )
              continue;
            break;
          }
LABEL_78:
          v126 |= v130;
LABEL_79:
          v6 = *(_QWORD *)(v4 + 32);
          v7 = (_BYTE *)(v6 + 40LL * (*(_DWORD *)(v4 + 40) & 0xFFFFFF));
        }
        v10 = v6 + 40LL * (unsigned int)sub_2E88FE0(v4);
        if ( v7 == (_BYTE *)v10 )
          goto LABEL_37;
        v12 = v10;
        v13 = 0;
        while ( 1 )
        {
LABEL_15:
          if ( *(_BYTE *)v12 )
            goto LABEL_14;
          if ( (*(_WORD *)(v12 + 2) & 0xFF0) != 0 )
            goto LABEL_14;
          v14 = *(_DWORD *)(v12 + 8);
          if ( v14 >= 0 )
            goto LABEL_14;
          v15 = *(_QWORD *)(a1 + 208);
          v16 = (_QWORD *)(*(_QWORD *)(v15 + 56) + 16LL * (v14 & 0x7FFFFFFF));
          if ( (*(_BYTE *)(v12 + 4) & 1) == 0 )
          {
            v17 = v16[1];
            if ( !v17 )
              goto LABEL_14;
            if ( (*(_BYTE *)(v17 + 3) & 0x10) == 0 )
            {
              v17 = *(_QWORD *)(v17 + 32);
              if ( !v17 || (*(_BYTE *)(v17 + 3) & 0x10) == 0 )
                goto LABEL_14;
            }
            v18 = *(_QWORD *)(v17 + 16);
            if ( *(_WORD *)(v18 + 68) != 10 )
              break;
          }
LABEL_26:
          v20 = sub_2EC06C0(v15, *v16 & 0xFFFFFFFFFFFFFFF8LL, byte_3F871B3, 0, v13, v11);
          v21 = *(_QWORD *)(v4 + 56);
          v22 = v20;
          v23 = *(_QWORD *)(*(_QWORD *)(a1 + 200) + 8LL);
          v149.m128i_i64[0] = v21;
          v24 = v23 - 440;
          if ( v21 )
          {
            v141 = v24;
            sub_B96E90((__int64)&v149, v21, 1);
            v24 = v141;
            v153 = (unsigned int *)v149.m128i_i64[0];
            if ( v149.m128i_i64[0] )
            {
              sub_B976B0((__int64)&v149, (unsigned __int8 *)v149.m128i_i64[0], (__int64)&v153);
              v24 = v141;
              v149.m128i_i64[0] = 0;
            }
          }
          else
          {
            v153 = 0;
          }
          v154 = 0;
          v155[0] = 0;
          sub_2F2A600(*(_QWORD *)(v4 + 24), v4, (__int64 *)&v153, v24, v22);
          if ( v153 )
            sub_B91220((__int64)&v153, (__int64)v153);
          if ( v149.m128i_i64[0] )
            sub_B91220((__int64)&v149, v149.m128i_i64[0]);
          sub_2EAB0C0(v12, v22);
          v25 = *(_BYTE *)(v12 + 4);
          if ( (v25 & 1) != 0 )
            *(_BYTE *)(v12 + 4) = v25 & 0xFE;
          v12 += 40;
          v13 = v144;
          if ( v7 == (_BYTE *)v12 )
          {
LABEL_36:
            v126 |= v13;
            goto LABEL_37;
          }
        }
        while ( 1 )
        {
          v17 = *(_QWORD *)(v17 + 32);
          if ( !v17 || (*(_BYTE *)(v17 + 3) & 0x10) == 0 )
            break;
          v19 = *(_QWORD *)(v17 + 16);
          if ( v19 != v18 )
          {
            v18 = *(_QWORD *)(v17 + 16);
            if ( *(_WORD *)(v19 + 68) == 10 )
              goto LABEL_26;
          }
        }
LABEL_14:
        v12 += 40;
        if ( v7 == (_BYTE *)v12 )
          goto LABEL_36;
        goto LABEL_15;
      }
LABEL_37:
      if ( (*(_BYTE *)v4 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v4 + 44) & 8) != 0 )
          v4 = *(_QWORD *)(v4 + 8);
      }
      v4 = *(_QWORD *)(v4 + 8);
      if ( v4 == v133 )
        return v126;
    }
    v30 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 200) + 16LL))(
            *(_QWORD *)(a1 + 200),
            *(_QWORD *)(v4 + 16),
            v28,
            *(_QWORD *)(a1 + 224),
            a2);
    v33 = sub_2EC06C0(*(_QWORD *)(a1 + 208), v30, byte_3F871B3, 0, v31, v32);
    v36 = *(_QWORD *)(a1 + 320) == 0;
    LODWORD(v148[0]) = v33;
    v37 = v33;
    if ( v36 )
    {
      v38 = *(unsigned int **)(a1 + 232);
      v39 = *(unsigned int *)(a1 + 240);
      v40 = &v38[v39];
      if ( v38 == v40 )
      {
        if ( v39 <= 7 )
        {
LABEL_61:
          v42 = v39 + 1;
          if ( v42 > *(unsigned int *)(a1 + 244) )
          {
            sub_C8D5F0(a1 + 232, (const void *)(a1 + 248), v42, 4u, v34, v35);
            v40 = (unsigned int *)(*(_QWORD *)(a1 + 232) + 4LL * *(unsigned int *)(a1 + 240));
          }
          *v40 = v37;
          ++*(_DWORD *)(a1 + 240);
          goto LABEL_136;
        }
        v110 = a1 + 280;
      }
      else
      {
        v41 = *(unsigned int **)(a1 + 232);
        while ( *v41 != v37 )
        {
          if ( v40 == ++v41 )
            goto LABEL_60;
        }
        if ( v40 != v41 )
          goto LABEL_136;
LABEL_60:
        if ( v39 <= 7 )
          goto LABEL_61;
        v143 = v4;
        v103 = *(unsigned int **)(a1 + 232);
        v145 = v29;
        v104 = (_QWORD *)(a1 + 288);
        do
        {
          v107 = sub_2DCC990((_QWORD *)(a1 + 280), (__int64)v104, v103);
          v109 = (_QWORD *)v108;
          if ( v108 )
          {
            v105 = v107 || (_QWORD *)v108 == v104 || *v103 < *(_DWORD *)(v108 + 32);
            v138 = v105;
            v106 = sub_22077B0(0x28u);
            *(_DWORD *)(v106 + 32) = *v103;
            sub_220F040(v138, v106, v109, v104);
            ++*(_QWORD *)(a1 + 320);
          }
          ++v103;
        }
        while ( v40 != v103 );
        v29 = v145;
        v4 = v143;
        v110 = a1 + 280;
      }
      *(_DWORD *)(a1 + 240) = 0;
      sub_2DCBF00(v110, (unsigned int *)v148);
    }
    else
    {
      sub_2DCBF00(a1 + 280, (unsigned int *)v148);
    }
LABEL_136:
    v100 = *(_QWORD *)(v4 + 56);
    v101 = *(_QWORD *)(*(_QWORD *)(a1 + 200) + 8LL);
    v149.m128i_i64[0] = v100;
    v102 = v101 - 400;
    if ( v100 )
    {
      sub_B96E90((__int64)&v149, v100, 1);
      v153 = (unsigned int *)v149.m128i_i64[0];
      if ( v149.m128i_i64[0] )
      {
        sub_B976B0((__int64)&v149, (unsigned __int8 *)v149.m128i_i64[0], (__int64)&v153);
        v149.m128i_i64[0] = 0;
      }
    }
    else
    {
      v153 = 0;
    }
    v154 = 0;
    v155[0] = 0;
    sub_2F26260(a3, (__int64 *)v4, (__int64 *)&v153, v102, (__int32)v148[0]);
    if ( v153 )
      sub_B91220((__int64)&v153, (__int64)v153);
    if ( v149.m128i_i64[0] )
      sub_B91220((__int64)&v149, v149.m128i_i64[0]);
    sub_2EAB0C0(v29, (int)v148[0]);
    v126 = 1;
    v6 = *(_QWORD *)(v4 + 32);
    goto LABEL_4;
  }
  return v126;
}

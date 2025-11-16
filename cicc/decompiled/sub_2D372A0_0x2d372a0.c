// Function: sub_2D372A0
// Address: 0x2d372a0
//
char __fastcall sub_2D372A0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // rdi
  __int64 v5; // rax
  _QWORD *v6; // r12
  _DWORD *v7; // rdi
  int v8; // edx
  _DWORD *v9; // rbx
  _DWORD *v10; // r12
  unsigned int v11; // ecx
  int v12; // eax
  unsigned __int64 v13; // rax
  unsigned int v14; // ecx
  __int64 v15; // rdi
  unsigned __int64 v16; // rdx
  unsigned int v17; // esi
  __int64 *v18; // rax
  __int64 v19; // r8
  __int64 v20; // rdx
  __int64 v21; // rbx
  __int64 v22; // r13
  __int64 v23; // r12
  __int64 v24; // rsi
  __int64 v25; // rbx
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  __m128i *v29; // rsi
  __m128i v30; // xmm1
  _QWORD *v31; // rdi
  unsigned int v32; // r13d
  __int64 v33; // rax
  int v34; // eax
  __int64 v35; // rax
  __int64 *v36; // rbx
  __m128i v37; // xmm3
  _QWORD *v38; // rdi
  unsigned int v39; // r13d
  __int64 v40; // rax
  int v41; // eax
  unsigned __int64 v42; // rdi
  __int64 v43; // rax
  unsigned __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rdi
  unsigned __int64 v47; // rsi
  unsigned int v48; // ecx
  __int64 *v49; // rax
  __int64 v50; // r8
  __int64 v51; // rdx
  __int64 v52; // r14
  __int64 v53; // r13
  __int64 v54; // rbx
  __int64 v55; // rsi
  __int64 v56; // rax
  unsigned int v57; // eax
  __int64 v58; // rax
  __int64 v59; // rax
  int v60; // eax
  unsigned int v61; // r8d
  __int64 *v62; // rax
  __int64 v63; // rcx
  __int64 v64; // rsi
  __int64 v65; // r10
  __int64 *v66; // r11
  unsigned int v67; // eax
  __int64 *v68; // rdx
  __int64 v69; // rdi
  unsigned int v70; // eax
  __int64 v71; // rax
  unsigned __int64 v72; // r8
  unsigned __int64 v73; // rdx
  unsigned __int64 v74; // rcx
  int v75; // esi
  __int64 v76; // rdi
  int v77; // eax
  int v78; // r10d
  __int64 v79; // rax
  unsigned __int64 v80; // r8
  unsigned __int64 v81; // rdx
  unsigned __int64 v82; // rcx
  int v83; // esi
  __int64 v84; // rdi
  __int64 v85; // rax
  __int64 v86; // rdi
  __int64 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // rbx
  __int64 v90; // r13
  unsigned __int64 v91; // rax
  __int64 v92; // rdx
  __int64 v93; // r8
  unsigned __int64 v94; // rcx
  unsigned int v95; // esi
  __int64 *v96; // rax
  __int64 v97; // rdi
  __int64 v98; // rdx
  __int64 v99; // rdi
  __int64 v100; // r12
  __int64 v101; // r14
  __int64 v102; // rsi
  __int64 v103; // rax
  char v104; // al
  int v105; // esi
  int v106; // esi
  unsigned int v107; // ecx
  __int64 v108; // rsi
  unsigned int v109; // edx
  __int64 *v110; // rax
  __int64 v111; // rdi
  __int64 v112; // rdx
  __int64 v113; // rbx
  __int64 *v114; // rdx
  __int64 *v115; // rax
  _QWORD *v116; // rax
  unsigned int *v117; // rax
  unsigned int v118; // r12d
  unsigned __int64 v119; // rax
  unsigned int v120; // edx
  __int64 v121; // rax
  __int64 *v122; // rax
  _QWORD *v123; // rdi
  char *v124; // rax
  unsigned __int8 v125; // al
  int v126; // eax
  int v127; // r10d
  unsigned int v128; // edx
  __int64 v129; // rbx
  __int64 v130; // rdx
  __int64 i; // rdx
  int v132; // eax
  int v133; // r10d
  int v134; // eax
  int j; // eax
  int v136; // r9d
  __int64 v138; // [rsp+20h] [rbp-150h]
  __int64 v139; // [rsp+28h] [rbp-148h]
  __int64 v140; // [rsp+30h] [rbp-140h]
  char v141; // [rsp+40h] [rbp-130h]
  __int64 v142; // [rsp+48h] [rbp-128h]
  __int64 v143; // [rsp+48h] [rbp-128h]
  __int64 v144; // [rsp+48h] [rbp-128h]
  __int64 v145; // [rsp+50h] [rbp-120h]
  __int64 v147; // [rsp+68h] [rbp-108h]
  __int64 *v148; // [rsp+68h] [rbp-108h]
  unsigned __int64 v149; // [rsp+68h] [rbp-108h]
  __int64 v150; // [rsp+68h] [rbp-108h]
  __int64 v151; // [rsp+70h] [rbp-100h]
  __int64 v152; // [rsp+70h] [rbp-100h]
  __int64 v153; // [rsp+70h] [rbp-100h]
  __int64 v154; // [rsp+70h] [rbp-100h]
  __int64 v155; // [rsp+70h] [rbp-100h]
  __int64 v156; // [rsp+78h] [rbp-F8h]
  int v157; // [rsp+78h] [rbp-F8h]
  _QWORD *v158; // [rsp+80h] [rbp-F0h]
  _QWORD *v159; // [rsp+88h] [rbp-E8h]
  __int64 v160; // [rsp+88h] [rbp-E8h]
  unsigned __int64 v161; // [rsp+98h] [rbp-D8h] BYREF
  __m128i v162; // [rsp+A0h] [rbp-D0h] BYREF
  __m128i v163; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v164; // [rsp+C0h] [rbp-B0h]
  __m128i v165; // [rsp+D0h] [rbp-A0h] BYREF
  __m128i v166; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v167; // [rsp+F0h] [rbp-80h]
  __m128i v168; // [rsp+100h] [rbp-70h] BYREF
  _QWORD v169[2]; // [rsp+110h] [rbp-60h] BYREF
  char v170; // [rsp+120h] [rbp-50h] BYREF

  v4 = *(_QWORD *)(a2 + 56);
  if ( v4 )
    v4 -= 24;
  LOBYTE(v5) = sub_B44020(v4);
  v158 = (_QWORD *)(a2 + 48);
  v159 = *(_QWORD **)(a2 + 56);
  if ( v159 != (_QWORD *)(a2 + 48) )
  {
    if ( !(_BYTE)v5 )
      goto LABEL_22;
    if ( !v159 )
    {
LABEL_69:
      if ( sub_B44020(0) )
        BUG();
LABEL_70:
      BUG();
    }
    while ( 1 )
    {
LABEL_6:
      if ( sub_B44020((__int64)(v159 - 3)) )
      {
        v86 = v159[5];
        if ( v86 )
        {
          v87 = sub_B14240(v86);
          v89 = v88;
          v90 = v87;
          if ( v88 != v87 )
          {
            while ( *(_BYTE *)(v90 + 32) )
            {
              v90 = *(_QWORD *)(v90 + 8);
              if ( v88 == v90 )
                goto LABEL_7;
            }
            if ( v88 != v90 )
            {
              do
              {
                v91 = sub_2D283A0(v90);
                v92 = *(unsigned int *)(a1 + 96);
                v93 = *(_QWORD *)(a1 + 80);
                v94 = v91;
                if ( (_DWORD)v92 )
                {
                  v95 = (v92 - 1) & (37 * v91);
                  v96 = (__int64 *)(v93 + 16LL * v95);
                  v97 = *v96;
                  if ( v94 == *v96 )
                  {
LABEL_166:
                    if ( v96 != (__int64 *)(v93 + 16 * v92) )
                    {
                      v98 = *(_QWORD *)(a1 + 104);
                      v99 = v98 + 56LL * *((unsigned int *)v96 + 2);
                      if ( v99 != v98 + 56LL * *(unsigned int *)(a1 + 112) )
                      {
                        v100 = *(_QWORD *)(v99 + 8);
                        v101 = v100 + 32LL * *(unsigned int *)(v99 + 16);
                        while ( v100 != v101 )
                        {
                          while ( 1 )
                          {
                            v102 = *(_QWORD *)(v101 - 16);
                            v101 -= 32;
                            if ( !v102 )
                              break;
                            sub_B91220(v101 + 16, v102);
                            if ( v100 == v101 )
                              goto LABEL_172;
                          }
                        }
LABEL_172:
                        *(_DWORD *)(v99 + 16) = 0;
                      }
                    }
                  }
                  else
                  {
                    v126 = 1;
                    while ( v97 != -4096 )
                    {
                      v127 = v126 + 1;
                      v95 = (v92 - 1) & (v126 + v95);
                      v96 = (__int64 *)(v93 + 16LL * v95);
                      v97 = *v96;
                      if ( v94 == *v96 )
                        goto LABEL_166;
                      v126 = v127;
                    }
                  }
                }
                v103 = sub_B11F60(v90 + 80);
                sub_AF47B0((__int64)&v168, *(unsigned __int64 **)(v103 + 16), *(unsigned __int64 **)(v103 + 24));
                if ( !LOBYTE(v169[0]) || v168.m128i_i64[0] )
                {
                  v104 = *(_BYTE *)(v90 + 64);
                  if ( v104 == 2 )
                  {
                    v168.m128i_i64[0] = a1;
                    v165.m128i_i64[0] = (__int64)a3;
                    v168.m128i_i64[1] = (__int64)&v165;
                    sub_2D36940(v168.m128i_i64, v90 & 0xFFFFFFFFFFFFFFF8LL);
                  }
                  else if ( v104 == 1 )
                  {
                    sub_2D36F30(a1, v90 | 4, a3);
                  }
                }
                do
                {
                  v90 = *(_QWORD *)(v90 + 8);
                  if ( v89 == v90 )
                    goto LABEL_7;
                }
                while ( *(_BYTE *)(v90 + 32) );
              }
              while ( v89 != v90 );
            }
          }
        }
      }
LABEL_7:
      if ( v158 != v159 )
      {
        v6 = v159;
        while ( 1 )
        {
          if ( !v6 )
            goto LABEL_70;
          v160 = (__int64)(v6 - 3);
          if ( *((_BYTE *)v6 - 24) != 85
            || (v43 = *(v6 - 7)) == 0
            || *(_BYTE *)v43
            || *(_QWORD *)(v43 + 24) != v6[7]
            || (*(_BYTE *)(v43 + 33) & 0x20) == 0
            || (unsigned int)(*(_DWORD *)(v43 + 36) - 68) > 3 )
          {
LABEL_11:
            v159 = v6;
            goto LABEL_12;
          }
          v44 = sub_2D283E0(v160);
          v45 = *(unsigned int *)(a1 + 96);
          v46 = *(_QWORD *)(a1 + 80);
          v47 = v44;
          if ( (_DWORD)v45 )
          {
            v48 = (v45 - 1) & (37 * v44);
            v49 = (__int64 *)(v46 + 16LL * v48);
            v50 = *v49;
            if ( v47 == *v49 )
            {
LABEL_81:
              if ( v49 != (__int64 *)(v46 + 16 * v45) )
              {
                v51 = *(_QWORD *)(a1 + 104);
                v52 = v51 + 56LL * *((unsigned int *)v49 + 2);
                if ( v52 != v51 + 56LL * *(unsigned int *)(a1 + 112) )
                {
                  v53 = *(_QWORD *)(v52 + 8);
                  v54 = v53 + 32LL * *(unsigned int *)(v52 + 16);
                  while ( v53 != v54 )
                  {
                    while ( 1 )
                    {
                      v55 = *(_QWORD *)(v54 - 16);
                      v54 -= 32;
                      if ( !v55 )
                        break;
                      sub_B91220(v54 + 16, v55);
                      if ( v53 == v54 )
                        goto LABEL_87;
                    }
                  }
LABEL_87:
                  *(_DWORD *)(v52 + 16) = 0;
                }
              }
            }
            else
            {
              v77 = 1;
              while ( v50 != -4096 )
              {
                v78 = v77 + 1;
                v48 = (v45 - 1) & (v77 + v48);
                v49 = (__int64 *)(v46 + 16LL * v48);
                v50 = *v49;
                if ( v47 == *v49 )
                  goto LABEL_81;
                v77 = v78;
              }
            }
          }
          v56 = *(v6 - 7);
          if ( !v56 || *(_BYTE *)v56 || *(_QWORD *)(v56 + 24) != v6[7] )
LABEL_260:
            BUG();
          v57 = *(_DWORD *)(v56 + 36);
          if ( v57 > 0x45 )
          {
            if ( v57 != 71 )
              goto LABEL_93;
          }
          else if ( v57 <= 0x43 )
          {
            goto LABEL_93;
          }
          v58 = *(_QWORD *)(v6[4 * (2LL - (*((_DWORD *)v6 - 5) & 0x7FFFFFF)) - 3] + 24LL);
          sub_AF47B0((__int64)&v168, *(unsigned __int64 **)(v58 + 16), *(unsigned __int64 **)(v58 + 24));
          if ( !LOBYTE(v169[0]) || v168.m128i_i64[0] )
          {
            v59 = *(v6 - 7);
            if ( !v59 || *(_BYTE *)v59 )
              BUG();
            if ( *(_QWORD *)(v59 + 24) != v6[7] )
              goto LABEL_260;
            v60 = *(_DWORD *)(v59 + 36);
            if ( v60 == 68 )
            {
              v168.m128i_i64[0] = a1;
              v165.m128i_i64[0] = (__int64)a3;
              v168.m128i_i64[1] = (__int64)&v165;
              sub_2D36D90(v168.m128i_i64, v160 & 0xFFFFFFFFFFFFFFF8LL);
            }
            else if ( v60 == 71 )
            {
              sub_2D36F30(a1, v160 & 0xFFFFFFFFFFFFFFFBLL, a3);
            }
          }
LABEL_93:
          v6 = (_QWORD *)v6[1];
          if ( v158 == v6 )
            goto LABEL_11;
        }
      }
      do
      {
        while ( 1 )
        {
LABEL_12:
          v7 = *(_DWORD **)(a1 + 224);
          v8 = *(_DWORD *)(a1 + 232);
          v9 = v7;
          v10 = &v7[*(unsigned int *)(a1 + 240)];
          if ( v8 && v7 != v10 )
          {
            while ( *v9 > 0xFFFFFFFD )
            {
              if ( ++v9 == v10 )
                goto LABEL_13;
            }
            if ( v9 == v10 )
            {
              ++*(_QWORD *)(a1 + 216);
              goto LABEL_15;
            }
            while ( 2 )
            {
              if ( !*(_DWORD *)(a3[25] + 4LL * (unsigned int)*v9) )
                goto LABEL_110;
              v61 = *(_DWORD *)(a1 + 272);
              v62 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 144) + 48LL) + 40LL * (unsigned int)(*v9 - 1));
              v63 = *v62;
              v64 = v62[4];
              v168.m128i_i64[0] = *v62;
              v168.m128i_i64[1] = v64;
              if ( !v61 )
              {
                ++*(_QWORD *)(a1 + 248);
                v165.m128i_i64[0] = 0;
LABEL_184:
                v105 = 2 * v61;
                goto LABEL_185;
              }
              v65 = *(_QWORD *)(a1 + 256);
              v66 = 0;
              v157 = 1;
              v67 = (v61 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4)
                      | ((unsigned __int64)(((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4))));
LABEL_118:
              v68 = (__int64 *)(v65 + 16LL * v67);
              v69 = *v68;
              if ( v63 == *v68 )
              {
                if ( v64 == v68[1] )
                  goto LABEL_110;
                if ( v69 == -4096 )
                  goto LABEL_127;
LABEL_120:
                if ( v69 == -8192 && v68[1] == -8192 && !v66 )
                  v66 = (__int64 *)(v65 + 16LL * v67);
              }
              else
              {
                if ( v69 != -4096 )
                  goto LABEL_120;
LABEL_127:
                if ( v68[1] == -4096 )
                {
                  v134 = *(_DWORD *)(a1 + 264);
                  if ( v66 )
                    v68 = v66;
                  ++*(_QWORD *)(a1 + 248);
                  v106 = v134 + 1;
                  v165.m128i_i64[0] = (__int64)v68;
                  if ( 4 * (v134 + 1) >= 3 * v61 )
                    goto LABEL_184;
                  if ( v61 - *(_DWORD *)(a1 + 268) - v106 <= v61 >> 3 )
                  {
                    v105 = v61;
LABEL_185:
                    sub_2D30530(a1 + 248, v105);
                    sub_2D2B5C0(a1 + 248, v168.m128i_i64, (__int64 **)&v165);
                    v63 = v168.m128i_i64[0];
                    v68 = (__int64 *)v165.m128i_i64[0];
                    v106 = *(_DWORD *)(a1 + 264) + 1;
                  }
                  *(_DWORD *)(a1 + 264) = v106;
                  if ( *v68 != -4096 || v68[1] != -4096 )
                    --*(_DWORD *)(a1 + 268);
                  *v68 = v63;
                  v68[1] = v168.m128i_i64[1];
LABEL_110:
                  if ( ++v9 == v10 )
                    goto LABEL_113;
                  while ( *v9 > 0xFFFFFFFD )
                  {
                    if ( v10 == ++v9 )
                      goto LABEL_113;
                  }
                  if ( v9 == v10 )
                  {
LABEL_113:
                    v8 = *(_DWORD *)(a1 + 232);
                    goto LABEL_13;
                  }
                  continue;
                }
              }
              break;
            }
            v70 = v157 + v67;
            ++v157;
            v67 = (v61 - 1) & v70;
            goto LABEL_118;
          }
LABEL_13:
          ++*(_QWORD *)(a1 + 216);
          if ( !v8 )
          {
            LODWORD(v5) = *(_DWORD *)(a1 + 236);
            if ( (_DWORD)v5 )
            {
              v5 = *(unsigned int *)(a1 + 240);
              if ( (unsigned int)v5 > 0x40 )
              {
                LOBYTE(v5) = sub_C7D6A0(*(_QWORD *)(a1 + 224), 4 * v5, 4);
                *(_DWORD *)(a1 + 240) = 0;
LABEL_232:
                *(_QWORD *)(a1 + 224) = 0;
              }
              else
              {
                v7 = *(_DWORD **)(a1 + 224);
LABEL_18:
                if ( 4LL * (unsigned int)v5 )
                  LOBYTE(v5) = (unsigned __int8)memset(v7, 255, 4LL * (unsigned int)v5);
              }
              *(_QWORD *)(a1 + 232) = 0;
              goto LABEL_21;
            }
            goto LABEL_21;
          }
          v7 = *(_DWORD **)(a1 + 224);
LABEL_15:
          v11 = 4 * v8;
          v5 = *(unsigned int *)(a1 + 240);
          if ( (unsigned int)(4 * v8) < 0x40 )
            v11 = 64;
          if ( v11 >= (unsigned int)v5 )
            goto LABEL_18;
          v128 = v8 - 1;
          if ( v128 )
          {
            _BitScanReverse(&v128, v128);
            v129 = (unsigned int)(1 << (33 - (v128 ^ 0x1F)));
            if ( (int)v129 < 64 )
              v129 = 64;
            if ( (_DWORD)v129 == (_DWORD)v5 )
            {
              *(_QWORD *)(a1 + 232) = 0;
              v5 = (__int64)&v7[v129];
              do
              {
                if ( v7 )
                  *v7 = -1;
                ++v7;
              }
              while ( (_DWORD *)v5 != v7 );
              goto LABEL_21;
            }
          }
          else
          {
            LODWORD(v129) = 64;
          }
          sub_C7D6A0((__int64)v7, 4 * v5, 4);
          LODWORD(v5) = sub_AF1560(4 * (int)v129 / 3u + 1);
          *(_DWORD *)(a1 + 240) = v5;
          if ( !(_DWORD)v5 )
            goto LABEL_232;
          v5 = sub_C7D670(4LL * (unsigned int)v5, 4);
          v130 = *(unsigned int *)(a1 + 240);
          *(_QWORD *)(a1 + 232) = 0;
          *(_QWORD *)(a1 + 224) = v5;
          for ( i = v5 + 4 * v130; i != v5; v5 += 4 )
          {
            if ( v5 )
              *(_DWORD *)v5 = -1;
          }
LABEL_21:
          if ( v158 == v159 )
            return v5;
LABEL_22:
          if ( !v159 )
            BUG();
          v12 = *((unsigned __int8 *)v159 - 24);
          if ( (_BYTE)v12 != 85 )
            break;
          v85 = *(v159 - 7);
          if ( !v85
            || *(_BYTE *)v85
            || *(_QWORD *)(v85 + 24) != v159[7]
            || (*(_BYTE *)(v85 + 33) & 0x20) == 0
            || (unsigned int)(*(_DWORD *)(v85 + 36) - 68) > 3 )
          {
            goto LABEL_25;
          }
          if ( v158 != v159 )
            goto LABEL_6;
        }
        LODWORD(v5) = v12 - 30;
        if ( (unsigned int)v5 <= 0xA )
          return v5;
LABEL_25:
        v156 = (__int64)(v159 - 3);
        v13 = sub_2D283E0((__int64)(v159 - 3));
        v14 = *(_DWORD *)(a1 + 96);
        v15 = *(_QWORD *)(a1 + 80);
        v16 = v13;
        if ( v14 )
        {
          v17 = (v14 - 1) & (37 * v13);
          v18 = (__int64 *)(v15 + 16LL * v17);
          v19 = *v18;
          if ( v16 == *v18 )
          {
LABEL_27:
            if ( v18 != (__int64 *)(v15 + 16LL * v14) )
            {
              v20 = *(_QWORD *)(a1 + 104);
              v21 = v20 + 56LL * *((unsigned int *)v18 + 2);
              if ( v21 != v20 + 56LL * *(unsigned int *)(a1 + 112) )
              {
                v22 = *(_QWORD *)(v21 + 8);
                v23 = v22 + 32LL * *(unsigned int *)(v21 + 16);
                while ( v22 != v23 )
                {
                  while ( 1 )
                  {
                    v24 = *(_QWORD *)(v23 - 16);
                    v23 -= 32;
                    if ( !v24 )
                      break;
                    sub_B91220(v23 + 16, v24);
                    if ( v22 == v23 )
                      goto LABEL_33;
                  }
                }
LABEL_33:
                *(_DWORD *)(v21 + 16) = 0;
              }
            }
          }
          else
          {
            v132 = 1;
            while ( v19 != -4096 )
            {
              v133 = v132 + 1;
              v17 = (v14 - 1) & (v132 + v17);
              v18 = (__int64 *)(v15 + 16LL * v17);
              v19 = *v18;
              if ( v16 == *v18 )
                goto LABEL_27;
              v132 = v133;
            }
          }
        }
        if ( (*((_BYTE *)v159 - 17) & 0x20) == 0 || !sub_B91C10(v156, 38) )
        {
          v107 = *(_DWORD *)(a1 + 64);
          if ( !v107 )
            goto LABEL_67;
          v108 = *(_QWORD *)(a1 + 48);
          v109 = (v107 - 1) & (((unsigned int)v156 >> 9) ^ ((unsigned int)v156 >> 4));
          v110 = (__int64 *)(v108 + ((unsigned __int64)v109 << 6));
          v111 = *v110;
          if ( v156 != *v110 )
          {
            for ( j = 1; ; j = v136 )
            {
              if ( v111 == -4096 )
                goto LABEL_67;
              v136 = j + 1;
              v109 = (v107 - 1) & (j + v109);
              v110 = (__int64 *)(v108 + ((unsigned __int64)v109 << 6));
              v111 = *v110;
              if ( v156 == *v110 )
                break;
            }
          }
          if ( v110 != (__int64 *)(v108 + ((unsigned __int64)v107 << 6)) )
          {
            v112 = v110[1];
            v140 = v112 + 40LL * *((unsigned int *)v110 + 4);
            if ( v112 != v140 )
            {
              v113 = v110[1];
              do
              {
                v118 = *(_DWORD *)v113;
                v145 = *(_QWORD *)(v113 + 8);
                v119 = *(_QWORD *)(v113 + 16);
                v120 = *(_DWORD *)v113;
                v168 = (__m128i)1uLL;
                v149 = v119;
                v169[0] = 0;
                sub_2D23D00(a1, a3, v120, &v168);
                v168 = (__m128i)1uLL;
                v169[0] = 0;
                sub_2D23C40(a1, a3, v118, &v168);
                sub_2D301F0(a1, a3, v118, 0);
                v121 = *(_QWORD *)(*(_QWORD *)(a1 + 144) + 48LL) + 40LL * (v118 - 1);
                v155 = *(_QWORD *)v121;
                v141 = *(_BYTE *)(v121 + 24);
                v139 = *(_QWORD *)(v121 + 8);
                v138 = *(_QWORD *)(v121 + 16);
                v144 = *(_QWORD *)(v121 + 32);
                v122 = (__int64 *)sub_BD5C60(v156);
                v123 = (_QWORD *)sub_B0D000(v122, 0, 0, 0, 1);
                if ( v141 )
                  v123 = (_QWORD *)sub_B0E470((__int64)v123, v138, v139);
                v124 = (char *)v169;
                v168.m128i_i64[1] = 0x300000000LL;
                v168.m128i_i64[0] = (__int64)v169;
                if ( v149 )
                {
                  v169[0] = 35;
                  v168.m128i_i32[2] = 2;
                  v169[1] = v149 >> 3;
                  v124 = &v170;
                }
                *(_QWORD *)v124 = 6;
                ++v168.m128i_i32[2];
                v150 = sub_B0D8A0(v123, (__int64)&v168, 0, 0);
                v161 = sub_2D283E0(v156);
                v125 = *(_BYTE *)(v155 - 16);
                if ( (v125 & 2) != 0 )
                  v114 = *(__int64 **)(v155 - 32);
                else
                  v114 = (__int64 *)(v155 - 16 - 8LL * ((v125 >> 2) & 0xF));
                v153 = *v114;
                v115 = (__int64 *)sub_B2BE50(*(_QWORD *)(a1 + 120));
                v116 = sub_B01860(v115, 0, 0, v153, v144, 0, 0, 1);
                v166 = 0u;
                v154 = (__int64)v116;
                v165.m128i_i32[0] = v118;
                v165.m128i_i64[1] = v150;
                v166.m128i_i64[1] = (__int64)sub_B98A20(v145, 0);
                sub_B10CB0(&v162, v154);
                if ( v166.m128i_i64[0] )
                  sub_B91220((__int64)&v166, v166.m128i_i64[0]);
                v166.m128i_i64[0] = v162.m128i_i64[0];
                if ( v162.m128i_i64[0] )
                  sub_B976B0((__int64)&v162, (unsigned __int8 *)v162.m128i_i64[0], (__int64)&v166);
                v117 = (unsigned int *)sub_2D363E0(a1 + 72, (__int64 *)&v161);
                sub_2D29B40(v117, (unsigned __int64)&v165);
                if ( v166.m128i_i64[0] )
                  sub_B91220((__int64)&v166, v166.m128i_i64[0]);
                if ( (_QWORD *)v168.m128i_i64[0] != v169 )
                  _libc_free(v168.m128i_u64[0]);
                v113 += 40;
              }
              while ( v140 != v113 );
            }
          }
          goto LABEL_67;
        }
        if ( (*((_BYTE *)v159 - 17) & 0x20) == 0 )
          goto LABEL_67;
        v151 = sub_B91C10(v156, 38);
        if ( v151 )
        {
          v25 = sub_AE94B0(v151);
          v151 = v26;
          if ( (*((_BYTE *)v159 - 17) & 0x20) == 0 )
            goto LABEL_219;
        }
        else
        {
          if ( (*((_BYTE *)v159 - 17) & 0x20) == 0 )
            goto LABEL_67;
          v25 = 0;
        }
        v27 = sub_B91C10(v156, 38);
        if ( !v27 )
        {
LABEL_219:
          v168.m128i_i64[0] = (__int64)v169;
          v168.m128i_i64[1] = 0x600000000LL;
          if ( v25 == v151 )
            goto LABEL_67;
          while ( 1 )
          {
LABEL_46:
            v147 = *(_QWORD *)(v25 + 24);
            sub_AF4850((__int64)&v162, v147);
            v30 = _mm_load_si128(&v163);
            v31 = *(_QWORD **)(a1 + 144);
            v165 = _mm_load_si128(&v162);
            v167 = v164;
            v166 = v30;
            v32 = sub_2D2C1F0(v31, &v165);
            v33 = 0;
            if ( (*((_BYTE *)v159 - 17) & 0x20) != 0 )
              v33 = sub_B91C10(v156, 38);
            v165.m128i_i32[0] = 0;
            v165.m128i_i64[1] = v33;
            v166.m128i_i64[0] = 0;
            sub_2D23D00(a1, a3, v32, &v165);
            if ( (unsigned __int8)sub_2D23AC0(a1, a3, 1, v32, v165.m128i_i32) )
            {
              sub_2D301F0(a1, a3, v32, 0);
              sub_2D36C00(a1, 0, v147, v156 & 0xFFFFFFFFFFFFFFFBLL);
              goto LABEL_45;
            }
            v34 = *(_DWORD *)(a3[25] + 4LL * v32);
            if ( v34 == 1 )
              break;
            if ( v34 == 2 )
            {
              sub_2D301F0(a1, a3, v32, 2);
              v25 = *(_QWORD *)(v25 + 8);
              if ( v25 == v151 )
              {
LABEL_52:
                v35 = v168.m128i_u32[2];
                goto LABEL_53;
              }
            }
            else
            {
              if ( !v34 )
              {
                v71 = a3[17] + 24LL * v32;
                if ( *(_DWORD *)v71 == 1 )
                {
                  sub_2D301F0(a1, a3, v32, 2);
                  v72 = v156 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_136:
                  v73 = v147;
                  v74 = v72;
                  v75 = 2;
                  v76 = a1;
LABEL_137:
                  sub_2D36C00(v76, v75, v73, v74);
                  goto LABEL_45;
                }
                v142 = *(_QWORD *)(v71 + 16);
                sub_2D301F0(a1, a3, v32, 1);
                v72 = v156 & 0xFFFFFFFFFFFFFFFBLL;
                v73 = v142 & 0xFFFFFFFFFFFFFFF8LL;
                if ( (v142 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                  goto LABEL_136;
                v74 = v156 & 0xFFFFFFFFFFFFFFFBLL;
                v75 = 1;
                v76 = a1;
                if ( (v142 & 4) == 0 )
                  goto LABEL_137;
                sub_2D367C0(a1, 1, v73, v156 & 0xFFFFFFFFFFFFFFFBLL);
              }
LABEL_45:
              v25 = *(_QWORD *)(v25 + 8);
              if ( v25 == v151 )
                goto LABEL_52;
            }
          }
          sub_2D301F0(a1, a3, v32, 1);
          goto LABEL_45;
        }
        v28 = *(_QWORD *)(v27 + 8);
        v29 = (__m128i *)(v28 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v28 & 4) == 0 )
          v29 = 0;
        sub_B967C0(&v168, v29);
        if ( v25 != v151 )
          goto LABEL_46;
        v35 = v168.m128i_u32[2];
        if ( v168.m128i_i32[2] )
        {
LABEL_53:
          v36 = (__int64 *)v168.m128i_i64[0];
          v148 = (__int64 *)(v168.m128i_i64[0] + 8 * v35);
          if ( (__int64 *)v168.m128i_i64[0] == v148 )
            goto LABEL_64;
          while ( 1 )
          {
            v152 = *v36;
            sub_AF48C0(&v162, *v36);
            v37 = _mm_load_si128(&v163);
            v38 = *(_QWORD **)(a1 + 144);
            v165 = _mm_load_si128(&v162);
            v167 = v164;
            v166 = v37;
            v39 = sub_2D2C1F0(v38, &v165);
            v40 = 0;
            if ( (*((_BYTE *)v159 - 17) & 0x20) != 0 )
              v40 = sub_B91C10(v156, 38);
            v165.m128i_i32[0] = 0;
            v165.m128i_i64[1] = v40;
            v166.m128i_i64[0] = 0;
            sub_2D23D00(a1, a3, v39, &v165);
            if ( (unsigned __int8)sub_2D23AC0(a1, a3, 1, v39, v165.m128i_i32) )
            {
              sub_2D301F0(a1, a3, v39, 0);
              sub_2D367C0(a1, 0, v152, v156 & 0xFFFFFFFFFFFFFFFBLL);
              goto LABEL_56;
            }
            v41 = *(_DWORD *)(a3[25] + 4LL * v39);
            if ( v41 == 1 )
              break;
            if ( v41 == 2 )
            {
              sub_2D301F0(a1, a3, v39, 2);
              if ( v148 == ++v36 )
              {
LABEL_63:
                v148 = (__int64 *)v168.m128i_i64[0];
LABEL_64:
                if ( v148 != v169 )
                {
                  v42 = (unsigned __int64)v148;
                  goto LABEL_66;
                }
                goto LABEL_67;
              }
            }
            else
            {
              if ( !v41 )
              {
                v79 = a3[17] + 24LL * v39;
                if ( *(_DWORD *)v79 == 1 )
                {
                  sub_2D301F0(a1, a3, v39, 2);
                  v80 = v156 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_149:
                  v81 = v152;
                  v82 = v80;
                  v83 = 2;
                  v84 = a1;
LABEL_150:
                  sub_2D367C0(v84, v83, v81, v82);
                  goto LABEL_56;
                }
                v143 = *(_QWORD *)(v79 + 16);
                sub_2D301F0(a1, a3, v39, 1);
                v80 = v156 & 0xFFFFFFFFFFFFFFFBLL;
                v81 = v143 & 0xFFFFFFFFFFFFFFF8LL;
                if ( (v143 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                  goto LABEL_149;
                v82 = v156 & 0xFFFFFFFFFFFFFFFBLL;
                v83 = 1;
                v84 = a1;
                if ( (v143 & 4) != 0 )
                  goto LABEL_150;
                sub_2D36C00(a1, 1, v81, v156 & 0xFFFFFFFFFFFFFFFBLL);
              }
LABEL_56:
              if ( v148 == ++v36 )
                goto LABEL_63;
            }
          }
          sub_2D301F0(a1, a3, v39, 1);
          goto LABEL_56;
        }
        v42 = v168.m128i_i64[0];
        if ( (_QWORD *)v168.m128i_i64[0] != v169 )
LABEL_66:
          _libc_free(v42);
LABEL_67:
        v159 = (_QWORD *)v159[1];
      }
      while ( v158 == v159 );
      if ( !v159 )
        goto LABEL_69;
    }
  }
  return v5;
}

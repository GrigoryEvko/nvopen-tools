// Function: sub_A10370
// Address: 0xa10370
//
__int64 __fastcall sub_A10370(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v6; // r14
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // r11
  __int64 v10; // r10
  __int64 v11; // r9
  unsigned __int32 v12; // r12d
  _BYTE *v13; // rdx
  char *v14; // rdx
  int v15; // edi
  char *v16; // r10
  int v17; // eax
  int v18; // edx
  void *v19; // rdi
  unsigned int v20; // eax
  unsigned int *v21; // rax
  __int64 v22; // rdx
  unsigned int *v23; // rdx
  char *v24; // rdx
  int v25; // edi
  char *v26; // r10
  int v27; // eax
  char *v28; // r12
  char *v29; // rbx
  unsigned int v30; // edx
  int v31; // edi
  unsigned int v32; // r14d
  __int64 v33; // rsi
  unsigned int v34; // edx
  int v35; // edx
  unsigned __int64 v36; // rax
  int v37; // ebx
  __int64 v38; // r12
  _DWORD *v39; // rax
  _DWORD *i; // rdx
  char v41; // al
  __int64 *k; // rdx
  __int64 *v43; // rbx
  __int64 *j; // r12
  __int8 *v45; // r11
  unsigned int v46; // edx
  char *v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 *m; // rax
  __int64 *v51; // rbx
  __int64 *v52; // r12
  __int64 v53; // r13
  __int64 v54; // rax
  __int64 *v55; // r12
  __int64 *v56; // rbx
  __int64 v57; // rdi
  __int8 v58; // dl
  unsigned __int32 v59; // eax
  __int64 v60; // rdx
  unsigned int v61; // eax
  __int64 *v62; // rbx
  __int64 *v63; // r13
  __int8 *v64; // rax
  __int8 *v65; // rdi
  __int64 v66; // rax
  __int64 *v67; // rbx
  __int64 v68; // r12
  __int64 *n; // r12
  __int64 v70; // rdi
  unsigned int v71; // eax
  __int8 v72; // dl
  __int32 *v73; // rbx
  __int64 v74; // r12
  __int32 *v75; // r12
  int v76; // edi
  unsigned int v77; // edx
  unsigned __int32 v78; // ecx
  int v79; // edi
  unsigned __int32 v80; // edx
  int v81; // eax
  __int64 v82; // rax
  __int64 v83; // rdx
  _BYTE *v84; // rdi
  unsigned __int32 v85; // eax
  unsigned __int64 *v86; // rdi
  size_t v87; // rdx
  __int64 jj; // r12
  __int64 v89; // r12
  __int64 v90; // rax
  __int64 v91; // rdx
  _QWORD *v92; // rax
  _QWORD *v93; // rdi
  _QWORD *v94; // rdx
  __int64 *v95; // rdx
  __int64 v96; // rax
  int v98; // r10d
  __int32 v99; // ecx
  unsigned int v100; // edx
  __int64 v101; // r11
  int v102; // edi
  char *v103; // rsi
  unsigned int v104; // r14d
  int v105; // edi
  __int8 *v106; // rbx
  __int64 v107; // rax
  __int32 v108; // ecx
  unsigned int v109; // edx
  __int64 v110; // r11
  __int32 v111; // ecx
  __int8 *v112; // rbx
  unsigned int v113; // ecx
  __int64 v114; // rsi
  unsigned int v115; // eax
  unsigned int v116; // ebx
  bool v117; // zf
  __int32 *v118; // rax
  __int64 v119; // rdx
  __int32 *v120; // rdx
  __int32 v121; // ecx
  int v122; // eax
  char *v123; // rdx
  unsigned int v124; // edx
  unsigned int v125; // ebx
  __int64 v126; // rax
  __int64 *v127; // rax
  __int8 v128; // al
  __int64 *v129; // rax
  __int64 v130; // rdx
  __int64 v131; // rdi
  __int64 v132; // rax
  __int64 v133; // rax
  __int8 v134; // al
  __int32 *v135; // rax
  __int64 v136; // rdx
  __int32 *ii; // rdx
  __int64 v138; // rdi
  __int64 v139; // rax
  int v140; // edi
  _QWORD *v141; // [rsp+0h] [rbp-80h]
  _QWORD *v142; // [rsp+0h] [rbp-80h]
  _QWORD *v143; // [rsp+0h] [rbp-80h]
  _QWORD *v144; // [rsp+0h] [rbp-80h]
  int v145; // [rsp+8h] [rbp-78h]
  int v146; // [rsp+8h] [rbp-78h]
  __int64 v147; // [rsp+8h] [rbp-78h]
  __int64 v148; // [rsp+8h] [rbp-78h]
  __int64 v149; // [rsp+8h] [rbp-78h]
  __int64 v150; // [rsp+8h] [rbp-78h]
  unsigned int v151; // [rsp+8h] [rbp-78h]
  _QWORD *v152; // [rsp+18h] [rbp-68h]
  __int64 v153; // [rsp+28h] [rbp-58h] BYREF
  __int64 v154; // [rsp+30h] [rbp-50h] BYREF
  void *s; // [rsp+38h] [rbp-48h]
  __int64 v156; // [rsp+40h] [rbp-40h]
  __int64 v157; // [rsp+48h] [rbp-38h]

  v6 = (_QWORD *)a2;
  v154 = 0;
  s = 0;
  v156 = 0;
  v157 = 0;
  while ( 2 )
  {
    v152 = v6;
    v7 = v6[2];
    v8 = v6[4];
    v9 = v6[6];
    v10 = v6[5] + 8LL;
LABEL_3:
    v11 = v10;
    while ( v9 != v7 )
    {
      v12 = *(_DWORD *)(v7 + 4);
      if ( v12 < a1->m128i_i32[2] && (v13 = *(_BYTE **)(a1->m128i_i64[0] + 8LL * v12)) != 0 )
      {
        if ( (unsigned __int8)(*v13 - 5) <= 0x1Fu && (v13[1] & 0x7F) == 2 )
        {
          a2 = (unsigned int)v157;
          if ( !(_DWORD)v157 )
          {
            ++v154;
            goto LABEL_173;
          }
          a5 = (unsigned int)(v157 - 1);
          a4 = (unsigned int)a5 & (37 * v12);
          v24 = (char *)s + 4 * a4;
          v25 = *(_DWORD *)v24;
          if ( v12 != *(_DWORD *)v24 )
          {
            v146 = 1;
            v26 = 0;
            while ( v25 != -1 )
            {
              if ( v25 == -2 && !v26 )
                v26 = v24;
              a4 = (unsigned int)a5 & (v146 + (_DWORD)a4);
              v24 = (char *)s + 4 * (unsigned int)a4;
              v25 = *(_DWORD *)v24;
              if ( v12 == *(_DWORD *)v24 )
                goto LABEL_9;
              ++v146;
            }
            if ( v26 )
              v24 = v26;
            ++v154;
            v27 = v156 + 1;
            if ( 4 * ((int)v156 + 1) < (unsigned int)(3 * v157) )
            {
              a4 = (unsigned int)(v157 - HIDWORD(v156) - v27);
              if ( (unsigned int)a4 <= (unsigned int)v157 >> 3 )
              {
                v144 = (_QWORD *)v11;
                v150 = v9;
                sub_A08C50((__int64)&v154, v157);
                if ( !(_DWORD)v157 )
                {
LABEL_387:
                  LODWORD(v156) = v156 + 1;
                  BUG();
                }
                a5 = (__int64)s;
                a4 = 0;
                v9 = v150;
                v104 = (v157 - 1) & (37 * v12);
                v11 = (__int64)v144;
                a2 = 1;
                v24 = (char *)s + 4 * v104;
                v105 = *(_DWORD *)v24;
                v27 = v156 + 1;
                if ( v12 != *(_DWORD *)v24 )
                {
                  while ( v105 != -1 )
                  {
                    if ( !a4 && v105 == -2 )
                      a4 = (__int64)v24;
                    v104 = (v157 - 1) & (a2 + v104);
                    v24 = (char *)s + 4 * v104;
                    v105 = *(_DWORD *)v24;
                    if ( v12 == *(_DWORD *)v24 )
                      goto LABEL_50;
                    a2 = (unsigned int)(a2 + 1);
                  }
                  if ( a4 )
                    v24 = (char *)a4;
                }
              }
              goto LABEL_50;
            }
LABEL_173:
            a2 = (unsigned int)(2 * v157);
            v143 = (_QWORD *)v11;
            v149 = v9;
            sub_A08C50((__int64)&v154, a2);
            if ( !(_DWORD)v157 )
              goto LABEL_387;
            v9 = v149;
            v11 = (__int64)v143;
            a4 = ((_DWORD)v157 - 1) & (37 * v12);
            v24 = (char *)s + 4 * a4;
            a5 = *(unsigned int *)v24;
            v27 = v156 + 1;
            if ( v12 != (_DWORD)a5 )
            {
              v76 = 1;
              a2 = 0;
              while ( (_DWORD)a5 != -1 )
              {
                if ( !a2 && (_DWORD)a5 == -2 )
                  a2 = (__int64)v24;
                a4 = ((_DWORD)v157 - 1) & (unsigned int)(v76 + a4);
                v24 = (char *)s + 4 * (unsigned int)a4;
                a5 = *(unsigned int *)v24;
                if ( v12 == (_DWORD)a5 )
                  goto LABEL_50;
                ++v76;
              }
              if ( a2 )
                v24 = (char *)a2;
            }
LABEL_50:
            LODWORD(v156) = v27;
            if ( *(_DWORD *)v24 != -1 )
              --HIDWORD(v156);
            *(_DWORD *)v24 = v12;
          }
        }
      }
      else
      {
        a2 = (unsigned int)v157;
        if ( !(_DWORD)v157 )
        {
          ++v154;
          goto LABEL_68;
        }
        a5 = (unsigned int)(v157 - 1);
        a4 = (unsigned int)a5 & (37 * v12);
        v14 = (char *)s + 4 * a4;
        v15 = *(_DWORD *)v14;
        if ( v12 != *(_DWORD *)v14 )
        {
          v145 = 1;
          v16 = 0;
          while ( v15 != -1 )
          {
            if ( v16 || v15 != -2 )
              v14 = v16;
            a4 = (unsigned int)a5 & (v145 + (_DWORD)a4);
            v15 = *((_DWORD *)s + (unsigned int)a4);
            if ( v12 == v15 )
              goto LABEL_9;
            ++v145;
            v16 = v14;
            v14 = (char *)s + 4 * (unsigned int)a4;
          }
          if ( !v16 )
            v16 = v14;
          ++v154;
          v17 = v156 + 1;
          if ( 4 * ((int)v156 + 1) < (unsigned int)(3 * v157) )
          {
            a4 = (unsigned int)v157 >> 3;
            if ( (int)v157 - HIDWORD(v156) - v17 <= (unsigned int)a4 )
            {
              v142 = (_QWORD *)v11;
              v148 = v9;
              sub_A08C50((__int64)&v154, v157);
              if ( !(_DWORD)v157 )
              {
LABEL_388:
                LODWORD(v156) = v156 + 1;
                BUG();
              }
              a4 = (unsigned int)(v157 - 1);
              v9 = v148;
              v32 = a4 & (37 * v12);
              v11 = (__int64)v142;
              v16 = (char *)s + 4 * v32;
              a2 = *(unsigned int *)v16;
              if ( v12 == (_DWORD)a2 )
              {
LABEL_77:
                v17 = v156 + 1;
              }
              else
              {
                v122 = 1;
                v123 = 0;
                while ( (_DWORD)a2 != -1 )
                {
                  if ( (_DWORD)a2 == -2 && !v123 )
                    v123 = v16;
                  a5 = (unsigned int)(v122 + 1);
                  v133 = (unsigned int)a4 & (v32 + v122);
                  v16 = (char *)s + 4 * v133;
                  v32 = v133;
                  a2 = *(unsigned int *)v16;
                  if ( v12 == (_DWORD)a2 )
                    goto LABEL_77;
                  v122 = a5;
                }
                v17 = v156 + 1;
                if ( v123 )
                  v16 = v123;
              }
            }
            goto LABEL_19;
          }
LABEL_68:
          a2 = (unsigned int)(2 * v157);
          v141 = (_QWORD *)v11;
          v147 = v9;
          sub_A08C50((__int64)&v154, a2);
          if ( !(_DWORD)v157 )
            goto LABEL_388;
          a5 = (__int64)s;
          v9 = v147;
          v11 = (__int64)v141;
          v30 = (v157 - 1) & (37 * v12);
          v16 = (char *)s + 4 * v30;
          v31 = *(_DWORD *)v16;
          v17 = v156 + 1;
          if ( v12 != *(_DWORD *)v16 )
          {
            a2 = 1;
            a4 = 0;
            while ( v31 != -1 )
            {
              if ( v31 == -2 && !a4 )
                a4 = (__int64)v16;
              v30 = (v157 - 1) & (a2 + v30);
              v151 = a2 + 1;
              a2 = v30;
              v16 = (char *)s + 4 * v30;
              v31 = *(_DWORD *)v16;
              if ( v12 == *(_DWORD *)v16 )
                goto LABEL_19;
              a2 = v151;
            }
            if ( a4 )
              v16 = (char *)a4;
          }
LABEL_19:
          LODWORD(v156) = v17;
          if ( *(_DWORD *)v16 != -1 )
            --HIDWORD(v156);
          *(_DWORD *)v16 = v12;
        }
      }
LABEL_9:
      v7 += 16;
      if ( v8 == v7 )
      {
        v7 = *(_QWORD *)v11;
        v10 = v11 + 8;
        v8 = *(_QWORD *)v11 + 512LL;
        goto LABEL_3;
      }
    }
    v18 = v156;
    v6 = v152;
    if ( (_DWORD)v156 )
    {
      v19 = s;
      v28 = (char *)s + 4 * (unsigned int)v157;
      v29 = (char *)s;
      if ( s != v28 )
      {
        while ( *(_DWORD *)v29 > 0xFFFFFFFD )
        {
          v29 += 4;
          if ( v28 == v29 )
            goto LABEL_24;
        }
        if ( v28 == v29 )
        {
          ++v154;
          goto LABEL_26;
        }
        do
        {
          a2 = *(unsigned int *)v29;
          v29 += 4;
          sub_A0FFA0(a1, a2, (__int64)v152, a4);
          if ( v29 == v28 )
            break;
          while ( *(_DWORD *)v29 > 0xFFFFFFFD )
          {
            v29 += 4;
            if ( v28 == v29 )
              goto LABEL_64;
          }
        }
        while ( v28 != v29 );
LABEL_64:
        v18 = v156;
      }
LABEL_24:
      ++v154;
      if ( !v18 )
      {
        if ( !HIDWORD(v156) )
          goto LABEL_32;
        v20 = v157;
        if ( (unsigned int)v157 > 0x40 )
        {
          a2 = 4LL * (unsigned int)v157;
          sub_C7D6A0(s, a2, 4);
          s = 0;
          v156 = 0;
          LODWORD(v157) = 0;
          goto LABEL_32;
        }
        v19 = s;
        goto LABEL_29;
      }
      v19 = s;
LABEL_26:
      a4 = (unsigned int)(4 * v18);
      v20 = v157;
      if ( (unsigned int)a4 < 0x40 )
        a4 = 64;
      if ( (unsigned int)a4 >= (unsigned int)v157 )
      {
LABEL_29:
        if ( 4LL * v20 )
        {
          a2 = 255;
          memset(v19, 255, 4LL * v20);
        }
        v156 = 0;
        goto LABEL_32;
      }
      a5 = (__int64)v19;
      v33 = 4LL * (unsigned int)v157;
      v34 = v18 - 1;
      if ( v34 )
      {
        _BitScanReverse(&v34, v34);
        a4 = 33 - (v34 ^ 0x1F);
        v35 = 1 << (33 - (v34 ^ 0x1F));
        if ( v35 < 64 )
          v35 = 64;
        if ( (_DWORD)v157 == v35 )
        {
          v156 = 0;
          a2 = (__int64)v19 + v33;
          do
          {
            if ( a5 )
              *(_DWORD *)a5 = -1;
            a5 += 4;
          }
          while ( a2 != a5 );
LABEL_32:
          if ( (unsigned __int32)a1[2].m128i_i32[0] >> 1 )
          {
            if ( (a1[2].m128i_i8[0] & 1) != 0 )
            {
LABEL_34:
              v21 = &a1[2].m128i_u32[2];
              v22 = 1;
              goto LABEL_35;
            }
            while ( 1 )
            {
              v21 = (unsigned int *)a1[2].m128i_i64[1];
              v22 = a1[3].m128i_u32[0];
LABEL_35:
              v23 = &v21[v22];
              a2 = *v21;
              if ( v23 != v21 )
              {
                while ( 1 )
                {
                  a2 = *v21;
                  a4 = (__int64)v21;
                  if ( (unsigned int)a2 <= 0xFFFFFFFD )
                    break;
                  if ( v23 == ++v21 )
                  {
                    a2 = *(unsigned int *)(a4 + 4);
                    break;
                  }
                }
              }
              sub_A0FFA0(a1, a2, (__int64)v152, a4);
              if ( !((unsigned __int32)a1[2].m128i_i32[0] >> 1) )
                break;
              if ( (a1[2].m128i_i8[0] & 1) != 0 )
                goto LABEL_34;
            }
          }
          continue;
        }
        v36 = (((4 * v35 / 3u + 1)
              | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)
              | (((4 * v35 / 3u + 1) | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 2)) >> 4)
            | (4 * v35 / 3u + 1)
            | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)
            | (((4 * v35 / 3u + 1) | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 2)
            | (((((4 * v35 / 3u + 1)
                | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)
                | (((4 * v35 / 3u + 1) | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 2)) >> 4)
              | (4 * v35 / 3u + 1)
              | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)
              | (((4 * v35 / 3u + 1) | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 2)) >> 8);
        v37 = (v36 | (v36 >> 16)) + 1;
        v38 = 4 * ((v36 | (v36 >> 16)) + 1);
      }
      else
      {
        v38 = 512;
        v37 = 128;
      }
      sub_C7D6A0(v19, v33, 4);
      a2 = 4;
      LODWORD(v157) = v37;
      v39 = (_DWORD *)sub_C7D670(v38, 4);
      v156 = 0;
      s = v39;
      for ( i = &v39[(unsigned int)v157]; i != v39; ++v39 )
      {
        if ( v39 )
          *v39 = -1;
      }
      goto LABEL_32;
    }
    break;
  }
  if ( (unsigned __int32)a1[2].m128i_i32[0] >> 1 )
    goto LABEL_24;
  v41 = a1[10].m128i_i8[0] & 1;
  k = (__int64 *)((unsigned __int32)a1[10].m128i_i32[0] >> 1);
  if ( (_DWORD)k )
  {
    if ( v41 )
    {
      v43 = &a1[10].m128i_i64[1];
      j = &a1[11].m128i_i64[1];
      goto LABEL_92;
    }
    v43 = (__int64 *)a1[10].m128i_i64[1];
    for ( j = &v43[2 * a1[11].m128i_u32[0]]; v43 != j; v43 += 2 )
    {
LABEL_92:
      if ( *v43 != -4096 && *v43 != -8192 )
        break;
    }
  }
  else
  {
    if ( v41 )
    {
      v106 = &a1[10].m128i_i8[8];
      v107 = 16;
    }
    else
    {
      v106 = (__int8 *)a1[10].m128i_i64[1];
      v107 = 16LL * a1[11].m128i_u32[0];
    }
    v43 = (__int64 *)&v106[v107];
    j = v43;
  }
  if ( j == v43 )
    goto LABEL_104;
  while ( 2 )
  {
    a2 = a1[8].m128i_i8[0] & 1;
    if ( (a1[8].m128i_i8[0] & 1) != 0 )
    {
      v45 = &a1[8].m128i_i8[8];
      v46 = 0;
      goto LABEL_98;
    }
    v77 = a1[9].m128i_u32[0];
    v45 = (__int8 *)a1[8].m128i_i64[1];
    if ( !v77 )
    {
      v78 = a1[8].m128i_u32[0];
      ++a1[7].m128i_i64[1];
      v47 = 0;
      v79 = (v78 >> 1) + 1;
      goto LABEL_190;
    }
    v46 = v77 - 1;
LABEL_98:
    a4 = v46 & (((unsigned int)*v43 >> 9) ^ ((unsigned int)*v43 >> 4));
    v47 = &v45[16 * a4];
    a5 = *(_QWORD *)v47;
    if ( *(_QWORD *)v47 == *v43 )
      goto LABEL_99;
    v98 = 1;
    v11 = 0;
    while ( 2 )
    {
      if ( a5 == -4096 )
      {
        v78 = a1[8].m128i_u32[0];
        if ( v11 )
          v47 = (char *)v11;
        ++a1[7].m128i_i64[1];
        v79 = (v78 >> 1) + 1;
        if ( (_BYTE)a2 )
        {
          a5 = (unsigned int)(4 * v79);
          v77 = 1;
          if ( (unsigned int)a5 >= 3 )
          {
LABEL_236:
            sub_A087A0((__int64)&a1[7].m128i_i64[1], 2 * v77);
            if ( (a1[8].m128i_i8[0] & 1) != 0 )
            {
              v11 = (__int64)&a1[8].m128i_i64[1];
              v99 = 0;
              goto LABEL_238;
            }
            v121 = a1[9].m128i_i32[0];
            v11 = a1[8].m128i_i64[1];
            if ( v121 )
            {
              v99 = v121 - 1;
LABEL_238:
              a5 = *v43;
              v100 = v99 & (((unsigned int)*v43 >> 9) ^ ((unsigned int)*v43 >> 4));
              v47 = (char *)(v11 + 16LL * v100);
              v101 = *(_QWORD *)v47;
              if ( *(_QWORD *)v47 != *v43 )
              {
                v102 = 1;
                v103 = 0;
                while ( v101 != -4096 )
                {
                  if ( v101 == -8192 && !v103 )
                    v103 = v47;
                  v100 = v99 & (v102 + v100);
                  v47 = (char *)(v11 + 16LL * v100);
                  v101 = *(_QWORD *)v47;
                  if ( a5 == *(_QWORD *)v47 )
                    goto LABEL_263;
                  ++v102;
                }
                goto LABEL_241;
              }
              goto LABEL_263;
            }
LABEL_386:
            a1[8].m128i_i32[0] = (2 * ((unsigned __int32)a1[8].m128i_i32[0] >> 1) + 2) | a1[8].m128i_i32[0] & 1;
            BUG();
          }
LABEL_191:
          if ( v77 - a1[8].m128i_i32[1] - v79 <= v77 >> 3 )
          {
            sub_A087A0((__int64)&a1[7].m128i_i64[1], v77);
            if ( (a1[8].m128i_i8[0] & 1) != 0 )
            {
              v11 = (__int64)&a1[8].m128i_i64[1];
              v108 = 0;
            }
            else
            {
              v111 = a1[9].m128i_i32[0];
              v11 = a1[8].m128i_i64[1];
              if ( !v111 )
                goto LABEL_386;
              v108 = v111 - 1;
            }
            a5 = *v43;
            v109 = v108 & (((unsigned int)*v43 >> 9) ^ ((unsigned int)*v43 >> 4));
            v47 = (char *)(v11 + 16LL * v109);
            v110 = *(_QWORD *)v47;
            if ( *(_QWORD *)v47 != *v43 )
            {
              v140 = 1;
              v103 = 0;
              while ( v110 != -4096 )
              {
                if ( v110 == -8192 && !v103 )
                  v103 = v47;
                v109 = v108 & (v140 + v109);
                v47 = (char *)(v11 + 16LL * v109);
                v110 = *(_QWORD *)v47;
                if ( a5 == *(_QWORD *)v47 )
                  goto LABEL_263;
                ++v140;
              }
LABEL_241:
              if ( v103 )
                v47 = v103;
            }
LABEL_263:
            v78 = a1[8].m128i_u32[0];
          }
          v80 = v78;
          a4 = v78 & 1;
          a2 = 2 * (v80 >> 1) + 2;
          a1[8].m128i_i32[0] = a2 | a4;
          if ( *(_QWORD *)v47 != -4096 )
            --a1[8].m128i_i32[1];
          *(_QWORD *)v47 = *v43;
          *((_QWORD *)v47 + 1) = v43[1];
          goto LABEL_99;
        }
        v77 = a1[9].m128i_u32[0];
LABEL_190:
        a5 = (unsigned int)(4 * v79);
        if ( 3 * v77 <= (unsigned int)a5 )
          goto LABEL_236;
        goto LABEL_191;
      }
      if ( !v11 && a5 == -8192 )
        v11 = (__int64)v47;
      a4 = v46 & (v98 + (_DWORD)a4);
      v47 = &v45[16 * (unsigned int)a4];
      a5 = *(_QWORD *)v47;
      if ( *v43 != *(_QWORD *)v47 )
      {
        ++v98;
        continue;
      }
      break;
    }
    do
    {
LABEL_99:
      v43 += 2;
      if ( v43 == j )
        goto LABEL_103;
    }
    while ( *v43 == -4096 || *v43 == -8192 );
    if ( j != v43 )
      continue;
    break;
  }
LABEL_103:
  k = (__int64 *)((unsigned __int32)a1[10].m128i_i32[0] >> 1);
LABEL_104:
  ++a1[9].m128i_i64[1];
  if ( (_DWORD)k )
  {
    if ( (a1[10].m128i_i8[0] & 1) == 0 )
    {
      a4 = (unsigned int)(4 * (_DWORD)k);
      goto LABEL_108;
    }
  }
  else
  {
    a2 = a1[10].m128i_u32[1];
    if ( !(_DWORD)a2 )
      goto LABEL_114;
    if ( (a1[10].m128i_i8[0] & 1) == 0 )
    {
      a4 = 0;
LABEL_108:
      v48 = a1[11].m128i_u32[0];
      if ( (unsigned int)v48 <= (unsigned int)a4 || (unsigned int)v48 <= 0x40 )
      {
        k = (__int64 *)a1[10].m128i_i64[1];
        v49 = 2 * v48;
        goto LABEL_111;
      }
      if ( !(_DWORD)k )
      {
        a2 = 16 * v48;
        sub_C7D6A0(a1[10].m128i_i64[1], 16 * v48, 8);
        a1[10].m128i_i8[0] |= 1u;
        goto LABEL_313;
      }
      v124 = (_DWORD)k - 1;
      if ( v124
        && (_BitScanReverse(&v124, v124), a4 = 33 - (v124 ^ 0x1F), v125 = 1 << (33 - (v124 ^ 0x1F)), v125 - 2 > 0x3D) )
      {
        if ( (_DWORD)v48 == v125 )
        {
          v117 = (a1[10].m128i_i64[0] & 1) == 0;
          a1[10].m128i_i64[0] &= 1uLL;
          if ( v117 )
          {
            k = (__int64 *)a1[10].m128i_i64[1];
            v126 = 2 * v48;
          }
          else
          {
            k = &a1[10].m128i_i64[1];
            v126 = 2;
          }
          v127 = &k[v126];
          do
          {
            if ( k )
              *k = -4096;
            k += 2;
          }
          while ( v127 != k );
          goto LABEL_114;
        }
        a2 = 16 * v48;
        sub_C7D6A0(a1[10].m128i_i64[1], 16 * v48, 8);
        v128 = a1[10].m128i_i8[0] | 1;
        a1[10].m128i_i8[0] = v128;
        if ( v125 <= 1 )
        {
LABEL_313:
          v117 = (a1[10].m128i_i64[0] & 1) == 0;
          a1[10].m128i_i64[0] &= 1uLL;
          if ( v117 )
          {
            v129 = (__int64 *)a1[10].m128i_i64[1];
            v130 = 2LL * a1[11].m128i_u32[0];
          }
          else
          {
            v129 = &a1[10].m128i_i64[1];
            v130 = 2;
          }
          for ( k = &v129[v130]; k != v129; v129 += 2 )
          {
            if ( v129 )
              *v129 = -4096;
          }
          goto LABEL_114;
        }
        v131 = 16LL * v125;
      }
      else
      {
        v125 = 64;
        sub_C7D6A0(a1[10].m128i_i64[1], 16 * v48, 8);
        v128 = a1[10].m128i_i8[0];
        v131 = 1024;
      }
      a2 = 8;
      a1[10].m128i_i8[0] = v128 & 0xFE;
      v132 = sub_C7D670(v131, 8);
      a1[11].m128i_i32[0] = v125;
      a1[10].m128i_i64[1] = v132;
      goto LABEL_313;
    }
  }
  k = &a1[10].m128i_i64[1];
  v49 = 2;
LABEL_111:
  for ( m = &k[v49]; m != k; k += 2 )
    *k = -4096;
  a1[10].m128i_i64[0] &= 1uLL;
LABEL_114:
  v51 = (__int64 *)a1[11].m128i_i64[1];
  v52 = &v51[2 * a1[12].m128i_u32[0]];
  if ( v51 != v52 )
  {
    do
    {
      while ( 1 )
      {
        a2 = *v51;
        v53 = v51[1];
        if ( *v51 && *(_BYTE *)a2 == 5 && (*(_BYTE *)(a2 + 1) & 0x7F) != 1 )
          a2 = sub_A06350((__int64)a1, a2);
        v54 = *(_QWORD *)(v53 + 8);
        if ( (v54 & 4) != 0 )
          break;
        v51 += 2;
        if ( v52 == v51 )
          goto LABEL_123;
      }
      v51 += 2;
      sub_BA6110(v54 & 0xFFFFFFFFFFFFFFF8LL, a2);
    }
    while ( v52 != v51 );
LABEL_123:
    v55 = (__int64 *)a1[11].m128i_i64[1];
    v56 = &v55[2 * a1[12].m128i_u32[0]];
    while ( v55 != v56 )
    {
      while ( 1 )
      {
        v57 = *(v56 - 1);
        v56 -= 2;
        if ( v57 )
          sub_BA65D0(v57, a2, k, a4, a5, v11);
        a2 = *v56;
        if ( !*v56 )
          break;
        sub_B91220(v56);
        if ( v55 == v56 )
          goto LABEL_129;
      }
    }
  }
LABEL_129:
  v58 = a1[6].m128i_i8[0];
  v59 = a1[6].m128i_u32[0];
  a1[12].m128i_i32[0] = 0;
  v60 = v58 & 1;
  v61 = v59 >> 1;
  if ( v61 )
  {
    if ( (_BYTE)v60 )
    {
      v62 = &a1[6].m128i_i64[1];
      v63 = &a1[7].m128i_i64[1];
      do
      {
LABEL_132:
        v60 = *v62;
        if ( *v62 != -8192 && v60 != -4096 )
          break;
        v62 += 2;
      }
      while ( v62 != v63 );
    }
    else
    {
      v62 = (__int64 *)a1[6].m128i_i64[1];
      v63 = &v62[2 * a1[7].m128i_u32[0]];
      if ( v62 != v63 )
        goto LABEL_132;
    }
  }
  else
  {
    if ( (_BYTE)v60 )
    {
      v112 = &a1[6].m128i_i8[8];
      v60 = 16;
    }
    else
    {
      v112 = (__int8 *)a1[6].m128i_i64[1];
      v60 = 16LL * a1[7].m128i_u32[0];
    }
    v62 = (__int64 *)&v112[v60];
    v63 = v62;
  }
  if ( v63 != v62 )
  {
LABEL_136:
    a2 = *v62;
    if ( (a1[8].m128i_i8[0] & 1) != 0 )
    {
      v64 = &a1[8].m128i_i8[8];
      v60 = 0;
      a4 = 0;
      v65 = &a1[8].m128i_i8[8];
LABEL_138:
      a5 = *(_QWORD *)v64;
      if ( *(_QWORD *)v64 == a2 )
      {
LABEL_139:
        a5 = *((_QWORD *)v64 + 1);
        if ( a5 )
        {
          v66 = *(_QWORD *)(v62[1] + 8);
          if ( (v66 & 4) != 0 )
          {
            a2 = a5;
            sub_BA6110(v66 & 0xFFFFFFFFFFFFFFF8LL, a5);
          }
          goto LABEL_142;
        }
      }
      else
      {
        v81 = 1;
        while ( a5 != -4096 )
        {
          v11 = (unsigned int)(v81 + 1);
          v60 = (unsigned int)a4 & (v81 + (_DWORD)v60);
          v64 = &v65[16 * (unsigned int)v60];
          a5 = *(_QWORD *)v64;
          if ( a2 == *(_QWORD *)v64 )
            goto LABEL_139;
          v81 = v11;
        }
      }
    }
    else
    {
      a4 = a1[9].m128i_u32[0];
      v65 = (__int8 *)a1[8].m128i_i64[1];
      if ( (_DWORD)a4 )
      {
        a4 = (unsigned int)(a4 - 1);
        v60 = (unsigned int)a4 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v64 = &v65[16 * v60];
        goto LABEL_138;
      }
    }
    v82 = *(_QWORD *)(v62[1] + 8);
    if ( (v82 & 4) != 0 )
      sub_BA6110(v82 & 0xFFFFFFFFFFFFFFF8LL, a2);
LABEL_142:
    while ( 1 )
    {
      v62 += 2;
      if ( v62 == v63 )
        break;
      if ( *v62 != -4096 && *v62 != -8192 )
      {
        if ( v63 != v62 )
          goto LABEL_136;
        break;
      }
    }
    v61 = (unsigned __int32)a1[6].m128i_i32[0] >> 1;
  }
  ++a1[5].m128i_i64[1];
  if ( v61 )
  {
    if ( (a1[6].m128i_i8[0] & 1) == 0 )
    {
      v61 *= 4;
      goto LABEL_150;
    }
LABEL_245:
    v67 = &a1[6].m128i_i64[1];
    v68 = 2;
LABEL_153:
    for ( n = &v67[v68]; n != v67; v67 += 2 )
    {
      if ( *v67 != -4096 )
      {
        if ( *v67 != -8192 )
        {
          v70 = v67[1];
          if ( v70 )
            sub_BA65D0(v70, a2, v60, a4, a5, v11);
        }
        *v67 = -4096;
      }
    }
    a1[6].m128i_i64[0] &= 1uLL;
    goto LABEL_161;
  }
  a4 = a1[6].m128i_u32[1];
  if ( !(_DWORD)a4 )
    goto LABEL_161;
  if ( (a1[6].m128i_i8[0] & 1) != 0 )
    goto LABEL_245;
LABEL_150:
  v60 = a1[7].m128i_u32[0];
  if ( v61 >= (unsigned int)v60 || (unsigned int)v60 <= 0x40 )
  {
    v67 = (__int64 *)a1[6].m128i_i64[1];
    v68 = 2LL * (unsigned int)v60;
    goto LABEL_153;
  }
  sub_A04FB0((__int64)&a1[5].m128i_i64[1]);
LABEL_161:
  v71 = (unsigned __int32)a1[4].m128i_i32[0] >> 1;
  if ( v71 )
  {
    v72 = a1[4].m128i_i8[0];
    if ( (v72 & 1) != 0 )
    {
      v73 = &a1[4].m128i_i32[2];
      v74 = 1;
    }
    else
    {
      v73 = (__int32 *)a1[4].m128i_i64[1];
      v74 = a1[5].m128i_u32[0];
    }
    v75 = &v73[v74];
    if ( v75 == v73 )
      goto LABEL_167;
    while ( (unsigned int)*v73 > 0xFFFFFFFD )
    {
      if ( v75 == ++v73 )
        goto LABEL_167;
    }
    if ( v75 == v73 )
    {
LABEL_167:
      ++a1[3].m128i_i64[1];
    }
    else
    {
      do
      {
        v83 = (unsigned int)*v73;
        v84 = *(_BYTE **)(a1->m128i_i64[0] + 8 * v83);
        if ( v84 && (unsigned __int8)(*v84 - 5) <= 0x1Fu )
          sub_B931A0(v84, a2, v83, a4, a5, v11);
        if ( ++v73 == v75 )
          break;
        while ( (unsigned int)*v73 > 0xFFFFFFFD )
        {
          if ( v75 == ++v73 )
            goto LABEL_208;
        }
      }
      while ( v75 != v73 );
LABEL_208:
      v85 = a1[4].m128i_u32[0];
      ++a1[3].m128i_i64[1];
      v71 = v85 >> 1;
      if ( !v71 )
      {
        if ( !a1[4].m128i_i32[1] )
          goto LABEL_214;
        if ( (a1[4].m128i_i8[0] & 1) != 0 )
          goto LABEL_211;
        v113 = 0;
LABEL_276:
        v114 = a1[5].m128i_u32[0];
        if ( v71 < (unsigned int)v114 && (unsigned int)v114 > 0x40 )
        {
          if ( !v113 )
          {
            sub_C7D6A0(a1[4].m128i_i64[1], 4 * v114, 4);
            a1[4].m128i_i8[0] |= 1u;
            goto LABEL_348;
          }
          if ( v113 == 1 || (_BitScanReverse(&v115, v113 - 1), v116 = 1 << (33 - (v115 ^ 0x1F)), v116 - 2 <= 0x3D) )
          {
            v116 = 64;
            sub_C7D6A0(a1[4].m128i_i64[1], 4 * v114, 4);
            v134 = a1[4].m128i_i8[0];
            v138 = 256;
          }
          else
          {
            if ( (_DWORD)v114 == v116 )
            {
              v117 = (a1[4].m128i_i64[0] & 1) == 0;
              a1[4].m128i_i64[0] &= 1uLL;
              if ( v117 )
              {
                v118 = (__int32 *)a1[4].m128i_i64[1];
                v119 = (unsigned int)v114;
              }
              else
              {
                v118 = &a1[4].m128i_i32[2];
                v119 = 1;
              }
              v120 = &v118[v119];
              do
              {
                if ( v118 )
                  *v118 = -1;
                ++v118;
              }
              while ( v120 != v118 );
              goto LABEL_214;
            }
            sub_C7D6A0(a1[4].m128i_i64[1], 4 * v114, 4);
            v134 = a1[4].m128i_i8[0] | 1;
            a1[4].m128i_i8[0] = v134;
            if ( v116 <= 1 )
            {
LABEL_348:
              v117 = (a1[4].m128i_i64[0] & 1) == 0;
              a1[4].m128i_i64[0] &= 1uLL;
              if ( v117 )
              {
                v135 = (__int32 *)a1[4].m128i_i64[1];
                v136 = a1[5].m128i_u32[0];
              }
              else
              {
                v135 = &a1[4].m128i_i32[2];
                v136 = 1;
              }
              for ( ii = &v135[v136]; ii != v135; ++v135 )
              {
                if ( v135 )
                  *v135 = -1;
              }
              goto LABEL_214;
            }
            v138 = 4LL * v116;
          }
          a1[4].m128i_i8[0] = v134 & 0xFE;
          v139 = sub_C7D670(v138, 4);
          a1[5].m128i_i32[0] = v116;
          a1[4].m128i_i64[1] = v139;
          goto LABEL_348;
        }
        v86 = (unsigned __int64 *)a1[4].m128i_i64[1];
        v87 = 4LL * (unsigned int)v114;
        if ( !v87 )
        {
LABEL_213:
          a1[4].m128i_i64[0] &= 1uLL;
          goto LABEL_214;
        }
LABEL_212:
        memset(v86, 255, v87);
        goto LABEL_213;
      }
      v72 = a1[4].m128i_i8[0];
    }
    if ( (v72 & 1) == 0 )
    {
      v113 = v71;
      v71 *= 4;
      goto LABEL_276;
    }
LABEL_211:
    v86 = &a1[4].m128i_u64[1];
    v87 = 4;
    goto LABEL_212;
  }
LABEL_214:
  for ( jj = v152[2]; jj != v152[6]; v152[2] = jj )
  {
    while ( 1 )
    {
      v90 = *(unsigned int *)(jj + 4);
      v91 = 0;
      if ( (unsigned int)v90 < a1->m128i_i32[2] )
        v91 = *(_QWORD *)(a1->m128i_i64[0] + 8 * v90);
      v92 = *(_QWORD **)(jj + 8);
      if ( v92 )
      {
        *v92 = v91;
        v93 = *(_QWORD **)(jj + 8);
        if ( *v93 )
          sub_B96E90(v93, *v93, 1);
        v153 = jj;
        sub_B91220(&v153);
      }
      v89 = v152[2];
      v94 = *(_QWORD **)(v89 + 8);
      if ( v89 != v152[4] - 16LL )
        break;
      if ( v94 )
        *v94 = 0;
      j_j___libc_free_0(v152[3], 512);
      v95 = (__int64 *)(v152[5] + 8LL);
      v152[5] = v95;
      jj = *v95;
      v96 = *v95 + 512;
      v152[3] = *v95;
      v152[4] = v96;
      v152[2] = jj;
      if ( jj == v152[6] )
        return sub_C7D6A0(s, 4LL * (unsigned int)v157, 4);
    }
    if ( v94 )
    {
      *v94 = 0;
      v89 = v152[2];
    }
    jj = v89 + 16;
  }
  return sub_C7D6A0(s, 4LL * (unsigned int)v157, 4);
}

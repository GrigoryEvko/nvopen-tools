// Function: sub_104E600
// Address: 0x104e600
//
__int64 __fastcall sub_104E600(__int64 a1)
{
  _BYTE *v2; // rsi
  char *v3; // rdi
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r8
  __int64 v8; // r9
  const __m128i *v9; // rcx
  const __m128i *v10; // rdx
  unsigned __int64 v11; // rbx
  __m128i *v12; // rax
  __int64 v13; // rcx
  __int64 i; // r8
  unsigned __int64 v15; // r9
  const __m128i *v16; // rax
  const __m128i *v17; // rcx
  unsigned __int64 v18; // rbx
  __int64 v19; // rax
  __m128i *v20; // rdi
  __m128i *v21; // rdx
  __m128i *v22; // rax
  __m128i *v23; // rcx
  __int64 result; // rax
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rdx
  unsigned int v28; // ecx
  __int64 *v29; // r13
  __int64 v30; // rsi
  __int64 v31; // rbx
  __int64 v32; // rcx
  __int64 v33; // rdi
  unsigned int v34; // ecx
  __int64 v35; // rsi
  unsigned int v36; // eax
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rsi
  _QWORD *v40; // rdx
  __int64 v41; // rcx
  unsigned int v42; // ecx
  unsigned __int64 v43; // rsi
  char v44; // al
  __int64 v45; // rdx
  int v46; // eax
  unsigned __int64 v47; // r11
  __int64 v48; // r10
  __int64 v49; // rdx
  unsigned int v50; // eax
  __int64 v51; // rsi
  __int64 v52; // rax
  _QWORD *v53; // rdx
  __int64 v54; // rcx
  int v55; // eax
  _BYTE *v56; // r9
  __int64 v57; // rbx
  unsigned int v58; // eax
  __int64 v59; // r11
  __int64 v60; // rdx
  unsigned int v61; // eax
  __int64 v62; // rax
  __int64 v63; // rdi
  __int64 v64; // rcx
  _QWORD *v65; // rdx
  __int64 v66; // rcx
  int v67; // eax
  unsigned __int64 v68; // r10
  unsigned int v69; // r10d
  __int8 v70; // si
  signed __int64 v71; // rsi
  __int64 v72; // rax
  __int64 v73; // rsi
  __int64 v74; // rdx
  _QWORD *v75; // rcx
  unsigned int v76; // eax
  int v77; // eax
  __int64 v78; // rcx
  __int64 v79; // rax
  __int64 v80; // rcx
  __int64 v81; // rax
  __int64 v82; // rsi
  __int64 v83; // rdx
  _QWORD *v84; // rcx
  unsigned int v85; // eax
  int v86; // eax
  __int64 v87; // rcx
  __int64 v88; // rax
  __int64 v89; // rcx
  __int64 v90; // rdx
  unsigned __int64 v91; // r9
  int v92; // eax
  int v93; // edx
  unsigned __int64 v94; // rdx
  unsigned __int64 v95; // r9
  int v96; // eax
  int v97; // edx
  unsigned __int64 v98; // rdx
  int v99; // eax
  int v100; // eax
  __int64 v101; // rdx
  unsigned __int64 v102; // r9
  int v103; // eax
  unsigned int v104; // r10d
  unsigned __int64 v105; // rbx
  __int64 v106; // rbx
  unsigned __int64 v107; // rbx
  unsigned __int64 v108; // rbx
  __int64 v109; // rbx
  _QWORD *v110; // rdx
  _QWORD *v111; // rsi
  _QWORD *v112; // rdx
  _QWORD *j; // rcx
  int v114; // ecx
  _QWORD *v115; // rdx
  _QWORD *k; // rcx
  int v117; // ecx
  char v118; // [rsp+17h] [rbp-289h]
  __int64 v119; // [rsp+20h] [rbp-280h]
  __int64 v120; // [rsp+20h] [rbp-280h]
  int v121; // [rsp+28h] [rbp-278h]
  unsigned __int64 v122; // [rsp+28h] [rbp-278h]
  _BYTE *v123; // [rsp+30h] [rbp-270h] BYREF
  __int64 v124; // [rsp+38h] [rbp-268h]
  _BYTE v125[48]; // [rsp+40h] [rbp-260h] BYREF
  unsigned int v126; // [rsp+70h] [rbp-230h]
  char v127[8]; // [rsp+80h] [rbp-220h] BYREF
  __int64 v128; // [rsp+88h] [rbp-218h]
  char v129; // [rsp+9Ch] [rbp-204h]
  _BYTE v130[64]; // [rsp+A0h] [rbp-200h] BYREF
  __m128i *v131; // [rsp+E0h] [rbp-1C0h]
  __int64 v132; // [rsp+E8h] [rbp-1B8h]
  __int8 *v133; // [rsp+F0h] [rbp-1B0h]
  char v134[8]; // [rsp+100h] [rbp-1A0h] BYREF
  __int64 v135; // [rsp+108h] [rbp-198h]
  char v136; // [rsp+11Ch] [rbp-184h]
  _BYTE v137[64]; // [rsp+120h] [rbp-180h] BYREF
  __m128i *v138; // [rsp+160h] [rbp-140h]
  __m128i *v139; // [rsp+168h] [rbp-138h]
  __int8 *v140; // [rsp+170h] [rbp-130h]
  _QWORD v141[3]; // [rsp+180h] [rbp-120h] BYREF
  char v142; // [rsp+19Ch] [rbp-104h]
  const __m128i *v143; // [rsp+1E0h] [rbp-C0h]
  const __m128i *v144; // [rsp+1E8h] [rbp-B8h]
  __int64 v145; // [rsp+1F0h] [rbp-B0h]
  char v146[8]; // [rsp+1F8h] [rbp-A8h] BYREF
  __int64 v147; // [rsp+200h] [rbp-A0h]
  char v148; // [rsp+214h] [rbp-8Ch]
  const __m128i *v149; // [rsp+258h] [rbp-48h]
  const __m128i *v150; // [rsp+260h] [rbp-40h]
  __int64 v151; // [rsp+268h] [rbp-38h]

  do
  {
    sub_104E270(v141, *(_QWORD *)a1);
    v2 = v130;
    v3 = v127;
    sub_C8CD80((__int64)v127, (__int64)v130, (__int64)v141, v4, v5, v6);
    v9 = v144;
    v10 = v143;
    v131 = 0;
    v132 = 0;
    v133 = 0;
    v11 = (char *)v144 - (char *)v143;
    if ( v144 == v143 )
    {
      v11 = 0;
      v12 = 0;
    }
    else
    {
      if ( v11 > 0x7FFFFFFFFFFFFFE0LL )
        goto LABEL_237;
      v12 = (__m128i *)sub_22077B0((char *)v144 - (char *)v143);
      v9 = v144;
      v10 = v143;
    }
    v131 = v12;
    v132 = (__int64)v12;
    v133 = &v12->m128i_i8[v11];
    if ( v9 == v10 )
    {
      v13 = (__int64)v12;
    }
    else
    {
      v13 = (__int64)v12->m128i_i64 + (char *)v9 - (char *)v10;
      do
      {
        if ( v12 )
        {
          *v12 = _mm_loadu_si128(v10);
          v12[1] = _mm_loadu_si128(v10 + 1);
        }
        v12 += 2;
        v10 += 2;
      }
      while ( v12 != (__m128i *)v13 );
    }
    v3 = v134;
    v132 = v13;
    v2 = v137;
    sub_C8CD80((__int64)v134, (__int64)v137, (__int64)v146, v13, v7, v8);
    v16 = v150;
    v17 = v149;
    v138 = 0;
    v139 = 0;
    v140 = 0;
    v18 = (char *)v150 - (char *)v149;
    if ( v150 == v149 )
    {
      v18 = 0;
      v20 = 0;
    }
    else
    {
      if ( v18 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_237:
        sub_4261EA(v3, v2, v10);
      v19 = sub_22077B0((char *)v150 - (char *)v149);
      v17 = v149;
      v20 = (__m128i *)v19;
      v16 = v150;
    }
    v138 = v20;
    v139 = v20;
    v140 = &v20->m128i_i8[v18];
    if ( v17 == v16 )
    {
      v22 = v20;
    }
    else
    {
      v21 = v20;
      v22 = (__m128i *)((char *)v20 + (char *)v16 - (char *)v17);
      do
      {
        if ( v21 )
        {
          *v21 = _mm_loadu_si128(v17);
          v21[1] = _mm_loadu_si128(v17 + 1);
        }
        v21 += 2;
        v17 += 2;
      }
      while ( v21 != v22 );
    }
    v139 = v22;
    v118 = 0;
    while ( 1 )
    {
      v23 = v131;
      result = (char *)v22 - (char *)v20;
      if ( v132 - (_QWORD)v131 != result )
        goto LABEL_19;
      if ( v131 == (__m128i *)v132 )
        break;
      result = (__int64)v20;
      while ( v23->m128i_i64[0] == *(_QWORD *)result )
      {
        v70 = v23[1].m128i_i8[8];
        if ( v70 != *(_BYTE *)(result + 24) || v70 && v23[1].m128i_i32[0] != *(_DWORD *)(result + 16) )
          break;
        v23 += 2;
        result += 32;
        if ( (__m128i *)v132 == v23 )
          goto LABEL_104;
      }
LABEL_19:
      v25 = *(unsigned int *)(a1 + 40);
      v26 = *(_QWORD *)(v132 - 32);
      v27 = *(_QWORD *)(a1 + 24);
      if ( (_DWORD)v25 )
      {
        i = (unsigned int)(v25 - 1);
        v28 = i & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v15 = 9LL * v28;
        v29 = (__int64 *)(v27 + 296LL * v28);
        v30 = *v29;
        if ( v26 == *v29 )
          goto LABEL_21;
        v15 = 1;
        while ( v30 != -4096 )
        {
          v104 = v15 + 1;
          v28 = i & (v15 + v28);
          v15 = 9LL * v28;
          v29 = (__int64 *)(v27 + 296LL * v28);
          v30 = *v29;
          if ( v26 == *v29 )
            goto LABEL_21;
          v15 = v104;
        }
      }
      v29 = (__int64 *)(v27 + 296LL * (unsigned int)v25);
LABEL_21:
      v126 = 0;
      v31 = *(_QWORD *)(v26 + 16);
      v123 = v125;
      v124 = 0x600000000LL;
      if ( v31 )
      {
        while ( 1 )
        {
          v32 = *(_QWORD *)(v31 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v32 - 30) <= 0xAu )
            break;
          v31 = *(_QWORD *)(v31 + 8);
          if ( !v31 )
            goto LABEL_33;
        }
LABEL_23:
        v33 = *(_QWORD *)(v32 + 40);
        if ( !(_DWORD)v25 )
          goto LABEL_29;
        v15 = (unsigned int)(v25 - 1);
        v34 = v15 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
        i = v27 + 296LL * v34;
        v35 = *(_QWORD *)i;
        if ( v33 != *(_QWORD *)i )
        {
          for ( i = 1; ; i = v69 )
          {
            if ( v35 == -4096 )
              goto LABEL_29;
            v69 = i + 1;
            v34 = v15 & (i + v34);
            i = v27 + 296LL * v34;
            v35 = *(_QWORD *)i;
            if ( v33 == *(_QWORD *)i )
              break;
          }
        }
        if ( i != v27 + 296 * v25 )
        {
          v36 = *(_DWORD *)(i + 288);
          if ( v126 < v36 )
          {
            if ( (v126 & 0x3F) != 0 )
              *(_QWORD *)&v123[8 * (unsigned int)v124 - 8] &= ~(-1LL << (v126 & 0x3F));
            v66 = (unsigned int)v124;
            v126 = v36;
            v15 = (v36 + 63) >> 6;
            if ( v15 != (unsigned int)v124 )
            {
              if ( v15 >= (unsigned int)v124 )
              {
                v68 = v15 - (unsigned int)v124;
                if ( v15 > HIDWORD(v124) )
                {
                  v120 = i;
                  v122 = v15 - (unsigned int)v124;
                  sub_C8D5F0((__int64)&v123, v125, v15, 8u, i, v15);
                  v66 = (unsigned int)v124;
                  i = v120;
                  v68 = v122;
                }
                if ( 8 * v68 )
                {
                  v119 = i;
                  v121 = v68;
                  memset(&v123[8 * v66], 0, 8 * v68);
                  LODWORD(v66) = v124;
                  i = v119;
                  LODWORD(v68) = v121;
                }
                LOBYTE(v36) = v126;
                LODWORD(v124) = v68 + v66;
              }
              else
              {
                LODWORD(v124) = (v36 + 63) >> 6;
              }
            }
            v67 = v36 & 0x3F;
            if ( v67 )
              *(_QWORD *)&v123[8 * (unsigned int)v124 - 8] &= ~(-1LL << v67);
          }
          v37 = 0;
          v38 = *(unsigned int *)(i + 232);
          v39 = 8 * v38;
          if ( (_DWORD)v38 )
          {
            do
            {
              v40 = &v123[v37];
              v41 = *(_QWORD *)(*(_QWORD *)(i + 224) + v37);
              v37 += 8;
              *v40 |= v41;
            }
            while ( v39 != v37 );
          }
        }
LABEL_29:
        while ( 1 )
        {
          v31 = *(_QWORD *)(v31 + 8);
          if ( !v31 )
            break;
          v32 = *(_QWORD *)(v31 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v32 - 30) <= 0xAu )
          {
            v27 = *(_QWORD *)(a1 + 24);
            v25 = *(unsigned int *)(a1 + 40);
            goto LABEL_23;
          }
        }
        v43 = (unsigned int)v124;
        if ( *(_DWORD *)(a1 + 8) != 1 || v126 )
          goto LABEL_40;
      }
      else
      {
LABEL_33:
        v42 = 0;
        if ( *(_DWORD *)(a1 + 8) != 1 )
          goto LABEL_51;
        v43 = v126;
      }
      v45 = (unsigned int)v43;
      v126 = *(_DWORD *)(a1 + 624);
      v44 = v126;
      v15 = (v126 + 63) >> 6;
      if ( v15 != (unsigned int)v43 )
      {
        if ( v15 >= (unsigned int)v43 )
        {
          v105 = v15 - (unsigned int)v43;
          if ( v15 > HIDWORD(v124) )
          {
            sub_C8D5F0((__int64)&v123, v125, (v126 + 63) >> 6, 8u, i, v15);
            v45 = (unsigned int)v124;
          }
          v15 = 8 * v105;
          if ( 8 * v105 )
          {
            memset(&v123[8 * v45], 255, 8 * v105);
            LODWORD(v45) = v124;
          }
          v44 = v126;
          LODWORD(v124) = v45 + v105;
          v43 = (unsigned int)(v45 + v105);
        }
        else
        {
          LODWORD(v124) = (v126 + 63) >> 6;
          v43 = (unsigned int)v124;
        }
      }
      v46 = v44 & 0x3F;
      if ( v46 )
      {
        *(_QWORD *)&v123[8 * v43 - 8] &= ~(-1LL << v46);
        v43 = (unsigned int)v124;
      }
LABEL_40:
      v47 = *((unsigned int *)v29 + 40);
      v42 = *((_DWORD *)v29 + 40);
      if ( (unsigned int)v43 <= (unsigned int)v47 )
        v42 = v43;
      if ( v42 )
      {
        v15 = (unsigned __int64)v123;
        v48 = v42;
        v49 = 0;
        while ( (*(_QWORD *)&v123[8 * v49] & ~*(_QWORD *)(v29[19] + 8 * v49)) == 0 )
        {
          v42 = ++v49;
          if ( v48 == v49 )
            goto LABEL_92;
        }
      }
      else
      {
LABEL_92:
        if ( v42 == (_DWORD)v43 )
          goto LABEL_51;
        while ( !*(_QWORD *)&v123[8 * v42] )
        {
          if ( ++v42 == (_DWORD)v43 )
            goto LABEL_51;
        }
      }
      v50 = v126;
      if ( *((_DWORD *)v29 + 54) < v126 )
      {
        v97 = v29[27] & 0x3F;
        if ( v97 )
        {
          *(_QWORD *)(v29[19] + 8 * v47 - 8) &= ~(-1LL << v97);
          v47 = *((unsigned int *)v29 + 40);
        }
        *((_DWORD *)v29 + 54) = v50;
        v98 = (v50 + 63) >> 6;
        if ( v98 != v47 )
        {
          if ( v98 >= v47 )
          {
            v106 = v98 - v47;
            if ( v98 > *((unsigned int *)v29 + 41) )
            {
              sub_C8D5F0((__int64)(v29 + 19), v29 + 21, v98, 8u, i, v15);
              v47 = *((unsigned int *)v29 + 40);
            }
            if ( 8 * v106 )
            {
              memset((void *)(v29[19] + 8 * v47), 0, 8 * v106);
              LODWORD(v47) = *((_DWORD *)v29 + 40);
            }
            v50 = *((_DWORD *)v29 + 54);
            *((_DWORD *)v29 + 40) = v106 + v47;
          }
          else
          {
            *((_DWORD *)v29 + 40) = (v50 + 63) >> 6;
          }
        }
        v99 = v50 & 0x3F;
        if ( v99 )
          *(_QWORD *)(v29[19] + 8LL * *((unsigned int *)v29 + 40) - 8) &= ~(-1LL << v99);
        v43 = (unsigned int)v124;
        if ( !(_DWORD)v124 )
        {
LABEL_172:
          v100 = *(_DWORD *)(a1 + 8);
          if ( v100 )
          {
            if ( v100 != 1 )
            {
              v56 = v123;
              goto LABEL_66;
            }
            goto LABEL_127;
          }
          goto LABEL_138;
        }
      }
      else if ( !(_DWORD)v43 )
      {
        goto LABEL_172;
      }
      v51 = 8 * v43;
      v52 = 0;
      do
      {
        v53 = (_QWORD *)(v52 + v29[19]);
        v54 = *(_QWORD *)&v123[v52];
        v52 += 8;
        *v53 |= v54;
      }
      while ( v51 != v52 );
      v42 = v124;
LABEL_51:
      v55 = *(_DWORD *)(a1 + 8);
      if ( v55 )
      {
        if ( v55 != 1 )
        {
          v43 = (unsigned __int64)v123;
          v56 = v123;
          goto LABEL_54;
        }
        if ( *((_DWORD *)v29 + 4) <= v42 )
          v42 = *((_DWORD *)v29 + 4);
        v72 = 0;
        v73 = 8LL * v42;
        if ( v42 )
        {
          do
          {
            v74 = *(_QWORD *)(v29[1] + v72);
            v75 = &v123[v72];
            v72 += 8;
            *v75 &= ~v74;
          }
          while ( v73 != v72 );
        }
LABEL_127:
        v76 = *((_DWORD *)v29 + 36);
        if ( v126 >= v76 )
          goto LABEL_128;
        if ( (v126 & 0x3F) != 0 )
          *(_QWORD *)&v123[8 * (unsigned int)v124 - 8] &= ~(-1LL << (v126 & 0x3F));
        v90 = (unsigned int)v124;
        v126 = v76;
        v91 = (v76 + 63) >> 6;
        if ( v91 != (unsigned int)v124 )
        {
          if ( v91 >= (unsigned int)v124 )
          {
            v108 = v91 - (unsigned int)v124;
            if ( v91 > HIDWORD(v124) )
            {
              sub_C8D5F0((__int64)&v123, v125, v91, 8u, i, v91);
              v90 = (unsigned int)v124;
            }
            v43 = (unsigned __int64)v123;
            v56 = v123;
            if ( 8 * v108 )
            {
              memset(&v123[8 * v90], 0, 8 * v108);
              v43 = (unsigned __int64)v123;
              LODWORD(v90) = v124;
              v56 = v123;
            }
            LOBYTE(v76) = v126;
            LODWORD(v124) = v108 + v90;
            goto LABEL_154;
          }
          LODWORD(v124) = (v76 + 63) >> 6;
        }
        v43 = (unsigned __int64)v123;
        v56 = v123;
LABEL_154:
        v92 = v76 & 0x3F;
        if ( !v92 )
          goto LABEL_129;
        *(_QWORD *)(v43 + 8LL * (unsigned int)v124 - 8) &= ~(-1LL << v92);
LABEL_128:
        v43 = (unsigned __int64)v123;
        v56 = v123;
LABEL_129:
        v77 = *((_DWORD *)v29 + 22);
        if ( !v77 )
          goto LABEL_185;
        v78 = (unsigned int)(v77 - 1);
        v79 = 0;
        v80 = 8 * v78;
        while ( 1 )
        {
          *(_QWORD *)&v56[v79] |= *(_QWORD *)(v29[10] + v79);
          if ( v80 == v79 )
            break;
          v56 = v123;
          v79 += 8;
        }
        v56 = v123;
        v42 = v124;
        v43 = (unsigned __int64)v123;
        goto LABEL_54;
      }
      if ( *((_DWORD *)v29 + 22) <= v42 )
        v42 = *((_DWORD *)v29 + 22);
      v81 = 0;
      v82 = 8LL * v42;
      if ( v42 )
      {
        do
        {
          v83 = *(_QWORD *)(v29[10] + v81);
          v84 = &v123[v81];
          v81 += 8;
          *v84 &= ~v83;
        }
        while ( v82 != v81 );
      }
LABEL_138:
      v85 = *((_DWORD *)v29 + 18);
      if ( v126 >= v85 )
        goto LABEL_139;
      if ( (v126 & 0x3F) != 0 )
        *(_QWORD *)&v123[8 * (unsigned int)v124 - 8] &= ~(-1LL << (v126 & 0x3F));
      v101 = (unsigned int)v124;
      v126 = v85;
      v102 = (v85 + 63) >> 6;
      if ( v102 != (unsigned int)v124 )
      {
        if ( v102 >= (unsigned int)v124 )
        {
          v107 = v102 - (unsigned int)v124;
          if ( v102 > HIDWORD(v124) )
          {
            sub_C8D5F0((__int64)&v123, v125, v102, 8u, i, v102);
            v101 = (unsigned int)v124;
          }
          v43 = (unsigned __int64)v123;
          v56 = v123;
          if ( 8 * v107 )
          {
            memset(&v123[8 * v101], 0, 8 * v107);
            v43 = (unsigned __int64)v123;
            LODWORD(v101) = v124;
            v56 = v123;
          }
          LOBYTE(v85) = v126;
          LODWORD(v124) = v107 + v101;
          goto LABEL_181;
        }
        LODWORD(v124) = (v85 + 63) >> 6;
      }
      v43 = (unsigned __int64)v123;
      v56 = v123;
LABEL_181:
      v103 = v85 & 0x3F;
      if ( !v103 )
        goto LABEL_140;
      *(_QWORD *)(v43 + 8LL * (unsigned int)v124 - 8) &= ~(-1LL << v103);
LABEL_139:
      v43 = (unsigned __int64)v123;
      v56 = v123;
LABEL_140:
      v86 = *((_DWORD *)v29 + 4);
      if ( !v86 )
      {
LABEL_185:
        v42 = v124;
        goto LABEL_54;
      }
      v87 = (unsigned int)(v86 - 1);
      v88 = 0;
      v89 = 8 * v87;
      while ( 1 )
      {
        *(_QWORD *)&v56[v88] |= *(_QWORD *)(v29[1] + v88);
        if ( v89 == v88 )
          break;
        v56 = v123;
        v88 += 8;
      }
      v56 = v123;
      v42 = v124;
      v43 = (unsigned __int64)v123;
LABEL_54:
      v57 = *((unsigned int *)v29 + 58);
      v58 = *((_DWORD *)v29 + 58);
      if ( v42 <= (unsigned int)v57 )
        v58 = v42;
      if ( v58 )
      {
        v59 = v58;
        v60 = 0;
        while ( (*(_QWORD *)(v43 + 8 * v60) & ~*(_QWORD *)(v29[28] + 8 * v60)) == 0 )
        {
          v58 = ++v60;
          if ( v59 == v60 )
            goto LABEL_71;
        }
LABEL_60:
        v61 = v126;
        if ( *((_DWORD *)v29 + 72) < v126 )
        {
          v93 = v29[36] & 0x3F;
          if ( v93 )
            *(_QWORD *)(v29[28] + 8 * v57 - 8) &= ~(-1LL << v93);
          v94 = *((unsigned int *)v29 + 58);
          *((_DWORD *)v29 + 72) = v61;
          v95 = (v61 + 63) >> 6;
          if ( v95 != v94 )
          {
            if ( v95 >= v94 )
            {
              v109 = v95 - v94;
              if ( v95 > *((unsigned int *)v29 + 59) )
              {
                sub_C8D5F0((__int64)(v29 + 28), v29 + 30, v95, 8u, i, v95);
                v94 = *((unsigned int *)v29 + 58);
              }
              if ( 8 * v109 )
              {
                memset((void *)(v29[28] + 8 * v94), 0, 8 * v109);
                LODWORD(v94) = *((_DWORD *)v29 + 58);
              }
              v61 = *((_DWORD *)v29 + 72);
              *((_DWORD *)v29 + 58) = v109 + v94;
            }
            else
            {
              *((_DWORD *)v29 + 58) = (v61 + 63) >> 6;
            }
          }
          v96 = v61 & 0x3F;
          if ( v96 )
            *(_QWORD *)(v29[28] + 8LL * *((unsigned int *)v29 + 58) - 8) &= ~(-1LL << v96);
          v43 = (unsigned __int64)v123;
        }
        v62 = 0;
        v63 = 8LL * (unsigned int)v124;
        if ( (_DWORD)v124 )
        {
          while ( 1 )
          {
            v64 = *(_QWORD *)(v43 + v62);
            v65 = (_QWORD *)(v62 + v29[28]);
            v62 += 8;
            *v65 |= v64;
            if ( v63 == v62 )
              break;
            v43 = (unsigned __int64)v123;
          }
          v118 = 1;
          v56 = v123;
        }
        else
        {
          v118 = 1;
          v56 = (_BYTE *)v43;
        }
      }
      else
      {
LABEL_71:
        while ( v42 != v58 )
        {
          if ( *(_QWORD *)(v43 + 8LL * v58) )
            goto LABEL_60;
          ++v58;
        }
      }
LABEL_66:
      if ( v56 != v125 )
        _libc_free(v56, v43);
      sub_104E0E0((__int64)v127);
      v20 = v138;
      v22 = v139;
    }
LABEL_104:
    v71 = v140 - (__int8 *)v20;
    if ( v20 )
      result = j_j___libc_free_0(v20, v71);
    if ( !v136 )
      result = _libc_free(v135, v71);
    if ( v131 )
    {
      v71 = v133 - (__int8 *)v131;
      result = j_j___libc_free_0(v131, v133 - (__int8 *)v131);
    }
    if ( !v129 )
      result = _libc_free(v128, v71);
    if ( v149 )
    {
      v71 = v151 - (_QWORD)v149;
      result = j_j___libc_free_0(v149, v151 - (_QWORD)v149);
    }
    if ( !v148 )
      result = _libc_free(v147, v71);
    if ( v143 )
    {
      v71 = v145 - (_QWORD)v143;
      result = j_j___libc_free_0(v143, v145 - (_QWORD)v143);
    }
    if ( !v142 )
      result = _libc_free(v141[1], v71);
  }
  while ( v118 );
  if ( *(_DWORD *)(a1 + 8) == 1 )
  {
    result = *(unsigned int *)(a1 + 32);
    if ( (_DWORD)result )
    {
      v110 = *(_QWORD **)(a1 + 24);
      result = 37LL * *(unsigned int *)(a1 + 40);
      v111 = &v110[37 * *(unsigned int *)(a1 + 40)];
      if ( v110 != v111 )
      {
        while ( 1 )
        {
          result = (__int64)v110;
          if ( *v110 != -8192 && *v110 != -4096 )
            break;
          v110 += 37;
          if ( v111 == v110 )
            return result;
        }
        if ( v110 != v111 )
        {
          do
          {
            v112 = *(_QWORD **)(result + 152);
            for ( j = &v112[*(unsigned int *)(result + 160)]; j != v112; ++v112 )
              *v112 = ~*v112;
            v114 = *(_DWORD *)(result + 216) & 0x3F;
            if ( v114 )
              *(_QWORD *)(*(_QWORD *)(result + 152) + 8LL * *(unsigned int *)(result + 160) - 8) &= ~(-1LL << v114);
            v115 = *(_QWORD **)(result + 224);
            for ( k = &v115[*(unsigned int *)(result + 232)]; v115 != k; ++v115 )
              *v115 = ~*v115;
            v117 = *(_DWORD *)(result + 288) & 0x3F;
            if ( v117 )
              *(_QWORD *)(*(_QWORD *)(result + 224) + 8LL * *(unsigned int *)(result + 232) - 8) &= ~(-1LL << v117);
            result += 296;
            if ( (_QWORD *)result == v111 )
              break;
            while ( *(_QWORD *)result == -4096 || *(_QWORD *)result == -8192 )
            {
              result += 296;
              if ( v111 == (_QWORD *)result )
                return result;
            }
          }
          while ( (_QWORD *)result != v111 );
        }
      }
    }
  }
  return result;
}

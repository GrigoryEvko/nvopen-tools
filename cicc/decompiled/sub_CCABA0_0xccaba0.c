// Function: sub_CCABA0
// Address: 0xccaba0
//
__int64 __fastcall sub_CCABA0(__int64 a1, char *a2, char *a3, unsigned int a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  unsigned int v7; // r10d
  char *v8; // r8
  bool v9; // r11
  __m128i *v10; // rcx
  bool v11; // r13
  __int64 v12; // r9
  __int64 v13; // rbx
  unsigned int v14; // esi
  unsigned int v15; // r13d
  unsigned int v16; // r14d
  __int64 v17; // r15
  __int64 v18; // r15
  _DWORD *v19; // rdi
  unsigned __int64 v20; // rsi
  int v21; // eax
  bool v22; // al
  __int64 v23; // r15
  __m128i *i; // rax
  __m128i *v25; // rax
  unsigned __int64 v26; // rax
  __m128i *v27; // rcx
  __int64 v28; // rsi
  __int64 *v29; // rbx
  __int64 v30; // rax
  __int64 *v31; // r12
  __int64 v32; // rsi
  __int64 *v33; // rax
  __int64 v34; // rcx
  unsigned __int64 v35; // rdx
  __int64 *v36; // rbx
  __int64 v37; // rcx
  __int64 v38; // rcx
  unsigned __int64 v39; // rdx
  __int64 v41; // rdx
  _DWORD *v42; // rax
  __m128i v43; // xmm1
  int v44; // eax
  int v45; // eax
  int v46; // edx
  unsigned __int64 v47; // rsi
  __int64 v48; // rdx
  int v49; // eax
  int v50; // eax
  __m128i *v51; // rax
  __int64 v52; // rsi
  __int64 v53; // rdi
  char *v54; // rax
  unsigned int v55; // edx
  __int64 *m128i_i64; // rax
  __int64 v57; // rcx
  __m128i *v58; // rcx
  unsigned __int64 v59; // rdx
  __int64 v60; // rax
  void *v61; // rdx
  bool v62; // zf
  __m128i *v63; // rdx
  __m128i *v64; // rax
  __m128i *v65; // rsi
  char v66; // r14
  __m128i *v67; // rax
  __int64 v68; // rdx
  unsigned int v69; // edi
  const char *v70; // r10
  __int64 v71; // rcx
  __m128i *v72; // rsi
  __m128i *v73; // rax
  __int64 v74; // r8
  __m128i v75; // xmm0
  __int64 v76; // rax
  char *v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // rdi
  char *v80; // rdx
  __m128i *v81; // rax
  __m128i *v82; // rcx
  int v83; // eax
  bool v84; // al
  _BOOL4 v85; // edx
  __m128i *v86; // rax
  __m128i *v87; // rcx
  __m128i *v88; // rax
  __m128i *j; // rdx
  unsigned __int64 v90; // r12
  unsigned __int64 v91; // rdx
  __m128i *v92; // rax
  char *v93; // rax
  __int64 v94; // rdx
  char *v95; // rax
  __m128i *v96; // rcx
  __int64 v97; // rdx
  _QWORD *v98; // rdi
  __int64 v99; // rsi
  __m128i *v100; // rax
  char *v101; // rax
  __int64 v102; // rdx
  __int64 v103; // rdi
  char *v104; // rsi
  size_t v105; // rdx
  __m128i v106; // [rsp+0h] [rbp-150h] BYREF
  __int64 v107; // [rsp+10h] [rbp-140h]
  unsigned int v108; // [rsp+1Ch] [rbp-134h]
  __m128i **v109; // [rsp+20h] [rbp-130h]
  __m128i *v110; // [rsp+28h] [rbp-128h]
  int v111; // [rsp+30h] [rbp-120h]
  int v112; // [rsp+34h] [rbp-11Ch]
  bool v113; // [rsp+3Ah] [rbp-116h]
  char v114; // [rsp+3Bh] [rbp-115h]
  int v115; // [rsp+3Ch] [rbp-114h]
  char *v116[3]; // [rsp+40h] [rbp-110h] BYREF
  char v117; // [rsp+5Ch] [rbp-F4h] BYREF
  bool v118; // [rsp+5Dh] [rbp-F3h]
  bool v119; // [rsp+5Eh] [rbp-F2h]
  char v120; // [rsp+5Fh] [rbp-F1h]
  void *dest; // [rsp+60h] [rbp-F0h]
  size_t v122; // [rsp+68h] [rbp-E8h]
  _QWORD v123[2]; // [rsp+70h] [rbp-E0h] BYREF
  _QWORD *v124; // [rsp+80h] [rbp-D0h] BYREF
  size_t n; // [rsp+88h] [rbp-C8h]
  _QWORD src[2]; // [rsp+90h] [rbp-C0h] BYREF
  void *v127[4]; // [rsp+A0h] [rbp-B0h] BYREF
  __int16 v128; // [rsp+C0h] [rbp-90h]
  __m128i *v129; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v130; // [rsp+D8h] [rbp-78h]
  _BYTE v131[112]; // [rsp+E0h] [rbp-70h] BYREF

  v116[0] = a2;
  v116[1] = a3;
  v108 = a4;
  v110 = (__m128i *)v131;
  v129 = (__m128i *)v131;
  v109 = &v129;
  v130 = 0x400000000LL;
  sub_C93960(v116, (__int64)&v129, 45, -1, 1, a6);
  v6 = v130;
  if ( (_DWORD)v130 )
  {
    v44 = sub_CC8470(v129->m128i_i64[0], v129->m128i_u64[1]);
    v7 = v130;
    if ( (unsigned int)v130 <= 1 )
    {
      v10 = v129;
      v112 = 0;
      LOBYTE(v12) = v44 != 0;
      v8 = 0;
      v9 = 0;
      v11 = 0;
      v115 = 0;
      v6 = 0;
      v111 = 0;
      v113 = 0;
      v114 = 0;
      goto LABEL_3;
    }
    v45 = sub_CC4230(v129[1].m128i_i64[0], v129[1].m128i_i64[1]);
    LOBYTE(v12) = v46 != 0;
    v111 = v45;
    v11 = v45 != 0;
    if ( v7 == 2 )
    {
      v113 = 0;
      v8 = 0;
      v9 = 0;
      v6 = 0;
      v112 = 0;
      v115 = 0;
      v114 = 0;
      goto LABEL_3;
    }
    v47 = v10[2].m128i_u64[1];
    v115 = sub_CC4400(v10[2].m128i_i64[0], v47);
    if ( v47 <= 5 )
    {
      v113 = 0;
      v114 = 0;
      if ( v47 != 5 )
      {
LABEL_70:
        v9 = v115 != 0;
        if ( v7 == 3 )
        {
          v112 = 0;
          v8 = 0;
          v6 = 0;
        }
        else
        {
          v112 = sub_CC4B20(v10[3].m128i_i64[0], v10[3].m128i_u64[1]);
          LOBYTE(v8) = v112 != 0;
          if ( v7 == 4 )
            v6 = 0;
          else
            v6 = sub_CC4070(v10[4].m128i_i64[0], v10[4].m128i_u64[1]);
        }
        goto LABEL_3;
      }
    }
    else
    {
      if ( *(_DWORD *)v48 != 2003269987 || (v49 = 0, *(_WORD *)(v48 + 4) != 28265) )
        v49 = 1;
      v113 = v49 == 0;
    }
    if ( *(_DWORD *)v48 != 1735289197 || (v50 = 0, *(_BYTE *)(v48 + 4) != 119) )
      v50 = 1;
    v114 = v50 == 0;
    goto LABEL_70;
  }
  v113 = 0;
  v7 = 0;
  v8 = 0;
  v9 = 0;
  v112 = 0;
  v10 = v129;
  v11 = 0;
  v12 = 0;
  v115 = 0;
  v111 = 0;
  v114 = 0;
LABEL_3:
  v13 = 0;
  v118 = v11;
  v14 = v7;
  v117 = v12;
  v15 = 0;
  v119 = v9;
  v120 = (char)v8;
  v107 = a1;
  if ( !(_BYTE)v12 )
    goto LABEL_6;
LABEL_4:
  if ( ++v13 == 4 )
    goto LABEL_18;
  do
  {
    v12 = (unsigned __int8)*(&v117 + v13);
    v15 = v13;
    if ( (_BYTE)v12 )
      goto LABEL_4;
LABEL_6:
    if ( !v14 )
      goto LABEL_4;
    v16 = 0;
    v17 = 0;
    while ( 1 )
    {
      if ( v16 <= 3 && *(&v117 + v16) )
        goto LABEL_50;
      v18 = v17;
      v19 = (_DWORD *)v10[v18].m128i_i64[0];
      v20 = v10[v18].m128i_u64[1];
      if ( v13 != 2 )
      {
        if ( v15 == 3 )
        {
          v112 = sub_CC4B20((__int64)v19, v20);
          if ( v112 )
            goto LABEL_15;
          v6 = sub_CC4070((__int64)v19, v20);
          v22 = v6 != 0;
        }
        else if ( v15 == 1 )
        {
          v111 = sub_CC4230((__int64)v19, v20);
          v22 = v111 != 0;
        }
        else
        {
          v21 = sub_CC8470(v19, v20);
          v10 = v129;
          v22 = v21 != 0;
        }
        if ( v22 )
          goto LABEL_15;
        goto LABEL_50;
      }
      v115 = sub_CC4400((__int64)v19, v20);
      if ( (unsigned __int64)v8 > 5 )
      {
        if ( *(_DWORD *)v41 != 2003269987 || (v83 = 0, *(_WORD *)(v41 + 4) != 28265) )
          v83 = 1;
        v84 = v83 == 0;
      }
      else
      {
        if ( v8 != (char *)5 )
        {
          v113 = 0;
          v114 = 0;
          if ( v115 )
            goto LABEL_15;
          goto LABEL_50;
        }
        v84 = 0;
      }
      v85 = *(_DWORD *)v41 != 1735289197 || *(_BYTE *)(v41 + 4) != 119;
      v113 = v84 || v115 != 0;
      if ( !v85 )
        break;
      if ( v113 )
      {
        v113 = v84;
        v114 = 0;
        goto LABEL_15;
      }
      v114 = 0;
      v115 = 0;
LABEL_50:
      v17 = v16 + 1;
      v16 = v17;
      if ( (_DWORD)v130 == (_DWORD)v17 )
      {
        v14 = v17;
        goto LABEL_4;
      }
    }
    if ( v84 || v115 != 0 )
    {
      v114 = v84 || v115 != 0;
      v113 = v84;
    }
    else
    {
      v115 = 0;
      v114 = 1;
    }
LABEL_15:
    if ( v16 > v15 )
    {
      v51 = &v10[v18];
      v52 = v10[v18].m128i_i64[1];
      v53 = v10[v18].m128i_i64[0];
      v51->m128i_i64[1] = 0;
      v51->m128i_i64[0] = (__int64)byte_3F871B3;
      if ( v52 )
      {
        v8 = &v117;
        if ( v15 <= 3 )
        {
LABEL_77:
          v54 = &v117 + v15;
          do
          {
            v55 = v15++;
            if ( !*v54 )
              goto LABEL_80;
            ++v54;
          }
          while ( v15 != 4 );
          v55 = 4;
          goto LABEL_80;
        }
        while ( 1 )
        {
          v55 = v15;
LABEL_80:
          v15 = v55 + 1;
          m128i_i64 = v129[v55].m128i_i64;
          v57 = m128i_i64[1];
          v12 = *m128i_i64;
          m128i_i64[1] = v52;
          *m128i_i64 = v53;
          if ( !v57 )
            break;
          v53 = v12;
          v52 = v57;
          if ( v15 <= 3 )
            goto LABEL_77;
        }
        v10 = v129;
        v14 = v130;
      }
      else
      {
        v10 = v129;
        v14 = v130;
        v8 = &v117;
      }
    }
    else
    {
      v14 = v130;
      v8 = &v117;
      if ( v16 < v15 )
      {
        v67 = v10;
LABEL_112:
        v68 = v16;
        v69 = v16;
        v70 = byte_3F871B3;
        v71 = 0;
        if ( v16 < v14 )
        {
          v72 = v67;
          while ( 1 )
          {
            v73 = &v72[v68];
            v74 = v73->m128i_i64[1];
            v75 = _mm_loadu_si128(v73);
            v73->m128i_i64[1] = v71;
            v12 = v73->m128i_i64[0];
            v73->m128i_i64[0] = (__int64)v70;
            if ( !v74 )
            {
              v67 = v129;
              v14 = v130;
              goto LABEL_122;
            }
            v76 = v69 + 1;
            v77 = &v117 + v76;
            while ( 1 )
            {
              v69 = v76;
              if ( (unsigned int)v76 > 3 )
                break;
              if ( !*v77++ )
                break;
              LODWORD(v76) = v76 + 1;
            }
            v78 = (unsigned int)v130;
            v68 = (unsigned int)v76;
            if ( (unsigned int)v76 >= (unsigned int)v130 )
              break;
            v72 = v129;
            v70 = (const char *)v12;
            v71 = v74;
          }
          if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
          {
            v106 = v75;
            sub_C8D5F0((__int64)v109, v110, (unsigned int)v130 + 1LL, 0x10u, v74, v12);
            v78 = (unsigned int)v130;
            v75 = _mm_load_si128(&v106);
          }
          v129[v78] = v75;
          v14 = v130 + 1;
          v67 = v129;
          LODWORD(v130) = v130 + 1;
        }
LABEL_122:
        v79 = v16 + 1;
        v80 = &v117 + v79;
        while ( 1 )
        {
          v16 = v79;
          if ( (unsigned int)v79 > 3 )
            break;
          if ( !*v80++ )
          {
            if ( (unsigned int)v79 < v15 )
              goto LABEL_112;
            break;
          }
          LODWORD(v79) = v79 + 1;
        }
        v10 = v67;
        v8 = &v117;
      }
    }
    *(&v117 + v13++) = 1;
  }
  while ( v13 != 4 );
LABEL_18:
  v23 = v107;
  if ( v117 )
  {
    if ( !v118 && !v119 )
    {
      if ( v120 )
      {
        if ( v10[1].m128i_i64[1] == 4 )
        {
          v42 = (_DWORD *)v10[1].m128i_i64[0];
          if ( *v42 == 1701736302 && !v10[2].m128i_i64[1] )
          {
            v43 = _mm_loadu_si128(v10 + 2);
            v10[2].m128i_i64[0] = (__int64)v42;
            v10[2].m128i_i64[1] = 4;
            v10[1] = v43;
          }
        }
      }
    }
  }
  for ( i = &v10[v14]; i != v10; ++v10 )
  {
    if ( !v10->m128i_i64[1] )
    {
      v10->m128i_i64[0] = (__int64)"unknown";
      v10->m128i_i64[1] = 7;
    }
  }
  v122 = 0;
  dest = v123;
  LOBYTE(v123[0]) = 0;
  if ( v112 != 17 )
  {
    if ( v111 == 12 && v112 == 5 )
    {
      v25 = v129;
      v129[3].m128i_i64[0] = (__int64)"gnueabihf";
      v25[3].m128i_i64[1] = 9;
    }
    goto LABEL_28;
  }
  v58 = v129;
  v59 = v129[3].m128i_u64[1];
  if ( v59 <= 0xA )
    goto LABEL_28;
  v60 = v129[3].m128i_i64[0];
  if ( *(_QWORD *)v60 != 0x6564696F72646E61LL || *(_WORD *)(v60 + 8) != 25185 || *(_BYTE *)(v60 + 10) != 105 )
    goto LABEL_28;
  v61 = (void *)(v59 - 11);
  if ( v61 )
  {
    v127[3] = v61;
    v127[0] = "android";
    v127[2] = (void *)(v60 + 11);
    v128 = 1283;
    sub_CA0F50((__int64 *)&v124, v127);
    v98 = dest;
    if ( v124 == src )
    {
      v105 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)dest = src[0];
        else
          memcpy(dest, src, n);
        v105 = n;
        v98 = dest;
      }
      v122 = v105;
      *((_BYTE *)v98 + v105) = 0;
      v98 = v124;
      goto LABEL_202;
    }
    if ( dest == v123 )
    {
      dest = v124;
      v122 = n;
      v123[0] = src[0];
    }
    else
    {
      v99 = v123[0];
      dest = v124;
      v122 = n;
      v123[0] = src[0];
      if ( v98 )
      {
        v124 = v98;
        src[0] = v99;
LABEL_202:
        n = 0;
        *(_BYTE *)v98 = 0;
        if ( v124 != src )
          j_j___libc_free_0(v124, src[0] + 1LL);
        v100 = v129;
        v129[3].m128i_i64[0] = (__int64)dest;
        v100[3].m128i_i64[1] = v122;
LABEL_28:
        v26 = (unsigned int)v130;
        if ( v115 != 14 )
          goto LABEL_29;
LABEL_91:
        if ( v26 != 4 )
        {
          if ( v26 <= 4 )
          {
            if ( HIDWORD(v130) <= 3 )
            {
              sub_C8D5F0((__int64)v109, v110, 4u, 0x10u, (__int64)v8, v12);
              v26 = (unsigned int)v130;
            }
            v63 = v129;
            v64 = &v129[v26];
            v65 = v129 + 4;
            v27 = v129;
            if ( v64 != &v129[4] )
            {
              do
              {
                if ( v64 )
                {
                  v64->m128i_i64[0] = 0;
                  v64->m128i_i64[1] = 0;
                }
                ++v64;
              }
              while ( v65 != v64 );
              v63 = v129;
              v27 = v129;
            }
            LODWORD(v130) = 4;
            goto LABEL_101;
          }
          LODWORD(v130) = 4;
        }
        v63 = v129;
        v27 = v129;
LABEL_101:
        v63[2].m128i_i64[1] = 7;
        v63[2].m128i_i64[0] = (__int64)"windows";
        if ( v112 )
        {
          v26 = (unsigned int)v130;
          if ( v6 <= 1 )
            goto LABEL_143;
        }
        else
        {
          v66 = v113 | v114;
          if ( v6 <= 1 )
          {
            v63[3].m128i_i64[1] = 4;
            v63[3].m128i_i64[0] = (__int64)"msvc";
            v26 = (unsigned int)v130;
            goto LABEL_32;
          }
          v101 = sub_CC6710(v6);
          v103 = v102;
          v63 = v129;
          v104 = v101;
          v26 = (unsigned int)v130;
          v129[3].m128i_i64[0] = (__int64)v104;
          v27 = v63;
          v63[3].m128i_i64[1] = v103;
          if ( !v66 )
            goto LABEL_32;
        }
        v13 = (unsigned int)v26;
        if ( (unsigned int)v26 == 5 )
        {
LABEL_185:
          v93 = sub_CC6710(v6);
          v27 = v129;
          v129[4].m128i_i64[0] = (__int64)v93;
          v26 = (unsigned int)v130;
          v27[4].m128i_i64[1] = v94;
          goto LABEL_32;
        }
        if ( (unsigned int)v26 > 5uLL )
        {
LABEL_184:
          LODWORD(v130) = 5;
          goto LABEL_185;
        }
        goto LABEL_163;
      }
    }
    v124 = src;
    v98 = src;
    goto LABEL_202;
  }
  v62 = v115 == 14;
  v129[3].m128i_i64[1] = 7;
  v58[3].m128i_i64[0] = (__int64)"android";
  v26 = (unsigned int)v130;
  if ( v62 )
    goto LABEL_91;
LABEL_29:
  if ( v114 )
  {
    if ( v26 != 4 )
    {
      if ( v26 <= 4 )
      {
        if ( HIDWORD(v130) <= 3 )
        {
          sub_C8D5F0((__int64)v109, v110, 4u, 0x10u, (__int64)v8, v12);
          v26 = (unsigned int)v130;
        }
        v63 = v129;
        v81 = &v129[v26];
        v82 = v129 + 4;
        if ( v81 != &v129[4] )
        {
          do
          {
            if ( v81 )
            {
              v81->m128i_i64[0] = 0;
              v81->m128i_i64[1] = 0;
            }
            ++v81;
          }
          while ( v82 != v81 );
          v63 = v129;
        }
        LODWORD(v130) = 4;
LABEL_142:
        v63[2].m128i_i64[1] = 7;
        v63[2].m128i_i64[0] = (__int64)"windows";
        v63[3].m128i_i64[0] = (__int64)"gnu";
        v26 = 4;
        v63[3].m128i_i64[1] = 3;
        if ( v6 <= 1 )
          goto LABEL_143;
LABEL_163:
        if ( HIDWORD(v130) <= 4 )
        {
          sub_C8D5F0((__int64)v109, v110, 5u, 0x10u, (__int64)v8, v12);
          v13 = (unsigned int)v130;
          v63 = v129;
        }
        v88 = &v63[v13];
        for ( j = v63 + 5; j != v88; ++v88 )
        {
          if ( v88 )
          {
            v88->m128i_i64[0] = 0;
            v88->m128i_i64[1] = 0;
          }
        }
        goto LABEL_184;
      }
      LODWORD(v130) = 4;
    }
    v63 = v129;
    goto LABEL_142;
  }
  if ( v113 )
  {
    if ( v26 != 4 )
    {
      if ( v26 <= 4 )
      {
        if ( HIDWORD(v130) <= 3 )
        {
          sub_C8D5F0((__int64)v109, v110, 4u, 0x10u, (__int64)v8, v12);
          v26 = (unsigned int)v130;
        }
        v63 = v129;
        v86 = &v129[v26];
        v87 = v129 + 4;
        if ( v86 != &v129[4] )
        {
          do
          {
            if ( v86 )
            {
              v86->m128i_i64[0] = 0;
              v86->m128i_i64[1] = 0;
            }
            ++v86;
          }
          while ( v87 != v86 );
          v63 = v129;
        }
        LODWORD(v130) = 4;
LABEL_162:
        v63[2].m128i_i64[1] = 7;
        v63[2].m128i_i64[0] = (__int64)"windows";
        v63[3].m128i_i64[0] = (__int64)"cygnus";
        v26 = 4;
        v63[3].m128i_i64[1] = 6;
        if ( v6 > 1 )
          goto LABEL_163;
LABEL_143:
        v27 = v63;
        goto LABEL_32;
      }
      LODWORD(v130) = 4;
    }
    v63 = v129;
    goto LABEL_162;
  }
  v27 = v129;
LABEL_32:
  if ( v27->m128i_i64[1] == 4 && *(_DWORD *)v27->m128i_i64[0] == 1818851428 )
  {
    if ( (unsigned int)v26 > 4 )
    {
      LODWORD(v130) = 4;
      if ( v115 == 38 )
        goto LABEL_187;
    }
    else
    {
      if ( v115 != 38 )
        goto LABEL_110;
LABEL_187:
      v95 = sub_CC5D50(v27[2].m128i_i64[0], v27[2].m128i_u64[1]);
      v96 = v129;
      v129->m128i_i64[0] = (__int64)v95;
      v96->m128i_i64[1] = v97;
    }
LABEL_110:
    v26 = (unsigned int)v130;
  }
  v28 = v108;
  if ( v108 - 3 <= 2 && v108 != v26 )
  {
    if ( v108 >= v26 )
    {
      v90 = v108 - v26;
      if ( v108 > (unsigned __int64)HIDWORD(v130) )
      {
        v28 = (__int64)v110;
        sub_C8D5F0((__int64)v109, v110, v108, 0x10u, (__int64)v8, v12);
        v26 = (unsigned int)v130;
      }
      v91 = v90;
      v92 = &v129[v26];
      do
      {
        if ( v92 )
        {
          v92->m128i_i64[0] = (__int64)"unknown";
          v92->m128i_i64[1] = 7;
        }
        ++v92;
        --v91;
      }
      while ( v91 );
      LODWORD(v130) = v90 + v130;
      v26 = (unsigned int)v130;
    }
    else
    {
      LODWORD(v130) = v108;
      v26 = v108;
    }
  }
  v29 = (__int64 *)v129;
  v30 = 2 * v26;
  *(_QWORD *)(v23 + 8) = 0;
  *(_QWORD *)v23 = v23 + 16;
  v31 = &v29[v30];
  *(_BYTE *)(v23 + 16) = 0;
  if ( v29 != &v29[v30] )
  {
    v32 = ((v30 * 8) >> 4) - 1;
    v33 = v29;
    do
    {
      v32 += v33[1];
      v33 += 2;
    }
    while ( v31 != v33 );
    sub_2240E30(v23, v32);
    v35 = v29[1];
    v28 = *v29;
    if ( v35 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(v23 + 8) )
      goto LABEL_222;
    v36 = v29 + 2;
    sub_2241490(v23, v28, v35, v34);
    if ( v36 != v31 )
    {
      while ( *(_QWORD *)(v23 + 8) != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490(v23, "-", 1, v37);
        v39 = v36[1];
        v28 = *v36;
        if ( v39 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(v23 + 8) )
          break;
        v36 += 2;
        sub_2241490(v23, v28, v39, v38);
        if ( v31 == v36 )
          goto LABEL_42;
      }
LABEL_222:
      sub_4262D8((__int64)"basic_string::append");
    }
  }
LABEL_42:
  if ( dest != v123 )
  {
    v28 = v123[0] + 1LL;
    j_j___libc_free_0(dest, v123[0] + 1LL);
  }
  if ( v129 != v110 )
    _libc_free(v129, v28);
  return v23;
}

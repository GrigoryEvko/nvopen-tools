// Function: sub_1895520
// Address: 0x1895520
//
__int64 __fastcall sub_1895520(_QWORD *a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // rax
  const __m128i *v7; // rsi
  __m128i *v8; // r14
  __m128i *v9; // r12
  __m128i *v10; // r14
  __m128i *v11; // r15
  __m128i *v12; // rbx
  char *v13; // rax
  unsigned __int64 *v14; // rdi
  char *v15; // rax
  bool v16; // zf
  __int64 v17; // rax
  _QWORD *v18; // r14
  void *v19; // r15
  unsigned __int64 v20; // r13
  _QWORD *v21; // rbx
  bool v22; // r12
  _QWORD *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  _QWORD *v26; // rdx
  __int64 v27; // rax
  bool v28; // r12
  unsigned int v29; // r12d
  __int64 v30; // rax
  __int64 v31; // rbx
  _QWORD *v32; // rbx
  __int64 v33; // rax
  int v34; // r14d
  __int64 v35; // rax
  _QWORD *v36; // rbx
  __int64 v37; // r12
  unsigned int v38; // edx
  unsigned int v39; // eax
  _QWORD *v40; // r12
  __int64 j; // rax
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v45; // rcx
  __int64 v46; // rax
  unsigned int v47; // eax
  _QWORD *v48; // rbx
  _QWORD *v49; // r12
  __int64 v50; // rsi
  __int64 v51; // rcx
  __int64 v52; // rax
  __int64 v53; // rax
  int v54; // edx
  __int64 v55; // rbx
  unsigned int v56; // eax
  _QWORD *v57; // r12
  unsigned __int64 v58; // rdx
  unsigned __int64 v59; // rax
  _QWORD *v60; // rax
  __int64 v61; // rdx
  _QWORD *v62; // rdx
  __int8 v63; // cl
  _QWORD *v64; // rbx
  __int8 v65; // al
  __int64 v66; // rax
  size_t n; // [rsp+38h] [rbp-118h]
  size_t na; // [rsp+38h] [rbp-118h]
  size_t nb; // [rsp+38h] [rbp-118h]
  __int64 v70; // [rsp+40h] [rbp-110h]
  _QWORD *v71; // [rsp+48h] [rbp-108h]
  _QWORD *v72; // [rsp+50h] [rbp-100h]
  _QWORD *v73; // [rsp+60h] [rbp-F0h]
  _QWORD *v74; // [rsp+68h] [rbp-E8h]
  __m128i *v75; // [rsp+70h] [rbp-E0h] BYREF
  __m128i *v76; // [rsp+78h] [rbp-D8h]
  const __m128i *v77; // [rsp+80h] [rbp-D0h]
  void *v78; // [rsp+90h] [rbp-C0h] BYREF
  _QWORD v79[2]; // [rsp+98h] [rbp-B8h] BYREF
  __int64 v80; // [rsp+A8h] [rbp-A8h]
  __int64 v81; // [rsp+B0h] [rbp-A0h]
  __m128i v82; // [rsp+C0h] [rbp-90h] BYREF
  char *v83; // [rsp+D0h] [rbp-80h]
  __int64 v84; // [rsp+D8h] [rbp-78h]
  __int64 i; // [rsp+E0h] [rbp-70h]
  int v86; // [rsp+E8h] [rbp-68h]
  __int64 v87; // [rsp+F0h] [rbp-60h]
  __int64 v88; // [rsp+F8h] [rbp-58h]
  __int64 v89; // [rsp+100h] [rbp-50h]
  int v90; // [rsp+108h] [rbp-48h]
  __int64 v91; // [rsp+110h] [rbp-40h]

  v3 = a2 + 24;
  v4 = *(_QWORD *)(a2 + 32);
  v75 = 0;
  v76 = 0;
  v77 = 0;
  if ( v4 == a2 + 24 )
  {
    v9 = 0;
    v8 = 0;
    sub_1893870(v82.m128i_i64, 0, 0);
  }
  else
  {
    do
    {
      v5 = v4 - 56;
      if ( !v4 )
        v5 = 0;
      if ( !sub_15E4F60(v5) && (*(_BYTE *)(v5 + 32) & 0xF) != 1 )
      {
        v6 = sub_1ACB5D0(v5);
        v82.m128i_i64[1] = v5;
        v7 = v76;
        v82.m128i_i64[0] = v6;
        if ( v76 == v77 )
        {
          sub_18936F0((const __m128i **)&v75, v76, &v82);
        }
        else
        {
          if ( v76 )
          {
            *v76 = _mm_loadu_si128(&v82);
            v7 = v76;
          }
          v76 = (__m128i *)&v7[1];
        }
      }
      v4 = *(_QWORD *)(v4 + 8);
    }
    while ( v3 != v4 );
    v8 = v76;
    v9 = v75;
    sub_1893870(v82.m128i_i64, v75, v76 - v75);
  }
  if ( v83 )
    sub_1895450(v9->m128i_i8, v8->m128i_i8, v83, v82.m128i_i64[1]);
  else
    sub_1890FA0(v9->m128i_i8, v8->m128i_i8);
  j_j___libc_free_0(v83, 16 * v82.m128i_i64[1]);
  v10 = v75;
  v11 = v76;
  if ( v75 != v76 )
  {
    v12 = v75;
    do
    {
      do
      {
        ++v12;
        if ( v10 != &v12[-1] && v12[-2].m128i_i64[0] == v12[-1].m128i_i64[0] )
          break;
        if ( v11 == v12 )
          goto LABEL_34;
      }
      while ( v12->m128i_i64[0] != v12[-1].m128i_i64[0] );
      v13 = (char *)v12[-1].m128i_i64[1];
      v82 = (__m128i)6uLL;
      v83 = v13;
      if ( v13 != 0 && v13 + 8 != 0 && v13 != (char *)-16LL )
        sub_164C220((__int64)&v82);
      v14 = (unsigned __int64 *)a1[32];
      if ( v14 == (unsigned __int64 *)a1[33] )
      {
        sub_1893930((unsigned __int64 **)a1 + 31, (char *)a1[32], &v82);
      }
      else
      {
        if ( v14 )
        {
          *v14 = 6;
          v14[1] = 0;
          v15 = v83;
          v16 = v83 == 0;
          v14[2] = (unsigned __int64)v83;
          if ( v15 + 8 != 0 && !v16 && v15 != (char *)-16LL )
            sub_1649AC0(v14, v82.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL);
          v14 = (unsigned __int64 *)a1[32];
        }
        a1[32] = v14 + 3;
      }
      if ( v83 != 0 && v83 + 8 != 0 && v83 != (char *)-16LL )
        sub_1649B30(&v82);
    }
    while ( v11 != v12 );
  }
LABEL_34:
  v72 = (_QWORD *)a1[31];
  v71 = a1 + 35;
  v74 = (_QWORD *)a1[32];
  do
  {
    v17 = a1[33];
    a1[31] = 0;
    a1[32] = 0;
    v70 = v17;
    a1[33] = 0;
    if ( v74 == v72 )
      goto LABEL_64;
    v73 = a1;
    v18 = v72;
    do
    {
      v19 = (void *)v18[2];
      if ( v19 && !sub_15E4F60(v18[2]) && (*((_BYTE *)v19 + 32) & 0xF) != 1 )
      {
        v20 = sub_1ACB5D0(v19);
        v21 = (_QWORD *)v73[36];
        if ( v21 )
        {
          while ( 1 )
          {
            if ( v20 == v21[5] )
            {
              v24 = v21[4];
              v82.m128i_i64[0] = (__int64)v19;
              v83 = 0;
              v25 = v73[34];
              v82.m128i_i64[1] = v24;
              v84 = 0;
              i = 0;
              v86 = 0;
              v87 = 0;
              v88 = 0;
              v89 = 0;
              v90 = 0;
              v91 = v25;
              v22 = (unsigned int)sub_1ACDA40(&v82) == -1;
              j___libc_free_0(v88);
              j___libc_free_0(v84);
            }
            else
            {
              v22 = v20 < v21[5];
            }
            v23 = (_QWORD *)v21[3];
            if ( v22 )
              v23 = (_QWORD *)v21[2];
            if ( !v23 )
              break;
            v21 = v23;
          }
          v26 = v21;
          if ( !v22 )
            goto LABEL_52;
        }
        else
        {
          v21 = v71;
        }
        if ( (_QWORD *)v73[37] == v21 )
        {
          v26 = v21;
          v29 = 1;
          if ( v21 == v71 )
          {
LABEL_57:
            n = (size_t)v26;
            v30 = sub_22077B0(48);
            *(_QWORD *)(v30 + 32) = v19;
            v31 = v30;
            *(_QWORD *)(v30 + 40) = v20;
            sub_220F040(v29, v30, n, v71);
            v78 = v19;
            ++v73[39];
            v79[0] = v31;
            sub_18923E0((__int64)&v82, (__int64)(v73 + 40), (__int64 *)&v78);
            goto LABEL_58;
          }
LABEL_96:
          if ( v20 == v26[5] )
          {
            v51 = v26[4];
            nb = (size_t)v26;
            v82.m128i_i64[0] = (__int64)v19;
            v52 = v73[34];
            v82.m128i_i64[1] = v51;
            v83 = 0;
            v84 = 0;
            i = 0;
            v86 = 0;
            v87 = 0;
            v88 = 0;
            v89 = 0;
            v90 = 0;
            v91 = v52;
            LOBYTE(v29) = (unsigned int)sub_1ACDA40(&v82) == -1;
            j___libc_free_0(v88);
            j___libc_free_0(v84);
            v26 = (_QWORD *)nb;
          }
          else
          {
            LOBYTE(v29) = v20 < v26[5];
          }
          v29 = (unsigned __int8)v29;
          goto LABEL_57;
        }
        v27 = sub_220EF80(v21);
        v26 = v21;
        v21 = (_QWORD *)v27;
LABEL_52:
        if ( v20 == v21[5] )
        {
          v45 = v21[4];
          na = (size_t)v26;
          v82.m128i_i64[1] = (__int64)v19;
          v46 = v73[34];
          v82.m128i_i64[0] = v45;
          v83 = 0;
          v84 = 0;
          i = 0;
          v86 = 0;
          v87 = 0;
          v88 = 0;
          v89 = 0;
          v90 = 0;
          v91 = v46;
          v28 = (unsigned int)sub_1ACDA40(&v82) == -1;
          j___libc_free_0(v88);
          j___libc_free_0(v84);
          v26 = (_QWORD *)na;
        }
        else
        {
          v28 = v20 > v21[5];
        }
        if ( !v28 )
          __asm { jmp     rax }
        if ( !v26 )
          BUG();
        v29 = 1;
        if ( v26 == v71 )
          goto LABEL_57;
        goto LABEL_96;
      }
LABEL_58:
      v18 += 3;
    }
    while ( v18 != v74 );
    a1 = v73;
    v32 = v72;
    do
    {
      v33 = v32[2];
      if ( v33 != 0 && v33 != -8 && v33 != -16 )
        sub_1649B30(v32);
      v32 += 3;
    }
    while ( v74 != v32 );
LABEL_64:
    if ( v72 )
      j_j___libc_free_0(v72, v70 - (_QWORD)v72);
    v74 = (_QWORD *)a1[32];
    v72 = (_QWORD *)a1[31];
  }
  while ( v74 != v72 );
  sub_1890C70(a1[36]);
  v34 = *((_DWORD *)a1 + 44);
  a1[36] = 0;
  ++a1[20];
  a1[37] = v71;
  a1[38] = v71;
  a1[39] = 0;
  if ( v34 || *((_DWORD *)a1 + 45) )
  {
    v35 = *((unsigned int *)a1 + 46);
    v36 = (_QWORD *)a1[21];
    v79[0] = 2;
    v79[1] = 0;
    v37 = 3 * v35;
    v38 = v35;
    v39 = 4 * v34;
    v80 = -8;
    v83 = 0;
    v40 = &v36[2 * v37];
    v81 = 0;
    if ( (unsigned int)(4 * v34) < 0x40 )
      v39 = 64;
    v84 = -16;
    v82.m128i_i64[1] = 2;
    i = 0;
    v78 = &unk_49F1DB8;
    v82.m128i_i64[0] = (__int64)&unk_49F1DB8;
    if ( v38 > v39 )
    {
      do
      {
        v53 = v36[3];
        *v36 = &unk_49EE2B0;
        if ( v53 != -8 && v53 != 0 && v53 != -16 )
          sub_1649B30(v36 + 1);
        v36 += 6;
      }
      while ( v36 != v40 );
      v82.m128i_i64[0] = (__int64)&unk_49EE2B0;
      if ( v84 != 0 && v84 != -8 && v84 != -16 )
        sub_1649B30(&v82.m128i_i64[1]);
      v78 = &unk_49EE2B0;
      if ( v80 != 0 && v80 != -8 && v80 != -16 )
        sub_1649B30(v79);
      v54 = *((_DWORD *)a1 + 46);
      if ( v34 )
      {
        v55 = 64;
        if ( v34 != 1 )
        {
          _BitScanReverse(&v56, v34 - 1);
          v55 = (unsigned int)(1 << (33 - (v56 ^ 0x1F)));
          if ( (int)v55 < 64 )
            v55 = 64;
        }
        v57 = (_QWORD *)a1[21];
        if ( (_DWORD)v55 == v54 )
        {
          a1[22] = 0;
          v82.m128i_i64[1] = 2;
          v83 = 0;
          v64 = &v57[6 * v55];
          v84 = -8;
          v82.m128i_i64[0] = (__int64)&unk_49F1DB8;
          i = 0;
          do
          {
            if ( v57 )
            {
              v65 = v82.m128i_i8[8];
              v57[2] = 0;
              v57[1] = v65 & 6;
              v66 = v84;
              v16 = v84 == 0;
              v57[3] = v84;
              if ( v66 != -8 && !v16 && v66 != -16 )
                sub_1649AC0(v57 + 1, v82.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL);
              *v57 = &unk_49F1DB8;
              v57[4] = i;
            }
            v57 += 6;
          }
          while ( v64 != v57 );
          v82.m128i_i64[0] = (__int64)&unk_49EE2B0;
          if ( v84 != -8 && v84 != 0 && v84 != -16 )
            sub_1649B30(&v82.m128i_i64[1]);
        }
        else
        {
          j___libc_free_0(a1[21]);
          v58 = ((((((((4 * (int)v55 / 3u + 1) | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 2)
                   | (4 * (int)v55 / 3u + 1)
                   | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 4)
                 | (((4 * (int)v55 / 3u + 1) | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v55 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 8)
               | (((((4 * (int)v55 / 3u + 1) | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v55 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v55 / 3u + 1) | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v55 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 16;
          v59 = (v58
               | (((((((4 * (int)v55 / 3u + 1) | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 2)
                   | (4 * (int)v55 / 3u + 1)
                   | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 4)
                 | (((4 * (int)v55 / 3u + 1) | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v55 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 8)
               | (((((4 * (int)v55 / 3u + 1) | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v55 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v55 / 3u + 1) | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v55 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1))
              + 1;
          *((_DWORD *)a1 + 46) = v59;
          v60 = (_QWORD *)sub_22077B0(48 * v59);
          v61 = *((unsigned int *)a1 + 46);
          a1[22] = 0;
          a1[21] = v60;
          v82.m128i_i64[1] = 2;
          v83 = 0;
          v62 = &v60[6 * v61];
          v84 = -8;
          v82.m128i_i64[0] = (__int64)&unk_49F1DB8;
          for ( i = 0; v62 != v60; v60 += 6 )
          {
            if ( v60 )
            {
              v63 = v82.m128i_i8[8];
              v60[2] = 0;
              v60[3] = -8;
              *v60 = &unk_49F1DB8;
              v60[1] = v63 & 6;
              v60[4] = i;
            }
          }
        }
      }
      else if ( v54 )
      {
        j___libc_free_0(a1[21]);
        a1[21] = 0;
        a1[22] = 0;
        *((_DWORD *)a1 + 46) = 0;
      }
      else
      {
        a1[22] = 0;
      }
    }
    else
    {
      if ( v36 == v40 )
      {
        a1[22] = 0;
      }
      else
      {
        for ( j = -8; ; j = v80 )
        {
          v42 = v36[3];
          if ( v42 != j )
          {
            if ( v42 != 0 && v42 != -8 && v42 != -16 )
            {
              sub_1649B30(v36 + 1);
              j = v80;
            }
            v36[3] = j;
            if ( j != 0 && j != -8 && j != -16 )
              sub_1649AC0(v36 + 1, v79[0] & 0xFFFFFFFFFFFFFFF8LL);
            v36[4] = v81;
          }
          v36 += 6;
          if ( v36 == v40 )
            break;
        }
        v43 = v84;
        a1[22] = 0;
        v82.m128i_i64[0] = (__int64)&unk_49EE2B0;
        if ( v43 != -8 && v43 != -16 && v43 )
          sub_1649B30(&v82.m128i_i64[1]);
      }
      v78 = &unk_49EE2B0;
      if ( v80 != 0 && v80 != -8 && v80 != -16 )
        sub_1649B30(v79);
    }
  }
  if ( *((_BYTE *)a1 + 224) )
  {
    v47 = *((_DWORD *)a1 + 54);
    if ( v47 )
    {
      v48 = (_QWORD *)a1[25];
      v49 = &v48[2 * v47];
      do
      {
        if ( *v48 != -4 && *v48 != -8 )
        {
          v50 = v48[1];
          if ( v50 )
            sub_161E7C0((__int64)(v48 + 1), v50);
        }
        v48 += 2;
      }
      while ( v49 != v48 );
    }
    j___libc_free_0(a1[25]);
    *((_BYTE *)a1 + 224) = 0;
  }
  if ( v75 )
    j_j___libc_free_0(v75, (char *)v77 - (char *)v75);
  return 0;
}

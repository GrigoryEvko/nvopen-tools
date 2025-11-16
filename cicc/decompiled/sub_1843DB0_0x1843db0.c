// Function: sub_1843DB0
// Address: 0x1843db0
//
void __fastcall sub_1843DB0(__int64 a1, __int64 a2)
{
  __int64 m128i_i64; // rsi
  int v3; // r8d
  int v4; // r9d
  __int64 v5; // rdx
  char v6; // al
  __int64 v7; // r15
  char *v8; // rax
  char *v9; // rdx
  __m128i *v10; // r13
  __m128i *p_src; // r9
  __m128i *v12; // r15
  __m128i *v13; // rax
  __int32 v14; // ebx
  __int64 v15; // r8
  __int64 v16; // r15
  char v17; // r13
  __int64 v18; // rdi
  unsigned __int64 v19; // rax
  unsigned int v20; // edx
  unsigned __int64 v21; // rax
  __int64 v22; // r13
  unsigned __int64 v23; // rax
  unsigned __int8 v24; // dl
  unsigned __int64 *v25; // rbx
  unsigned __int64 *v26; // r12
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  int v29; // eax
  char v30; // cl
  __int64 v31; // rbx
  _QWORD *v32; // rax
  int v33; // r8d
  int v34; // r9d
  __int64 v35; // r15
  __int32 v36; // r12d
  const void *v37; // r9
  __m128i *v38; // r14
  __int64 v39; // rdi
  __int64 v40; // r8
  __int64 v41; // r15
  __int64 v42; // rcx
  int v43; // edx
  __int64 v44; // r14
  __int64 v45; // rbx
  __int32 v46; // r12d
  __int64 v47; // r15
  int v48; // edx
  __int64 v49; // r15
  unsigned __int64 *v50; // rbx
  __int64 n; // [rsp+10h] [rbp-310h]
  unsigned __int64 v52; // [rsp+18h] [rbp-308h]
  char v53; // [rsp+26h] [rbp-2FAh]
  char v54; // [rsp+27h] [rbp-2F9h]
  __int64 v56; // [rsp+30h] [rbp-2F0h]
  unsigned int v58; // [rsp+40h] [rbp-2E0h]
  unsigned int v59; // [rsp+44h] [rbp-2DCh]
  __int64 v60; // [rsp+48h] [rbp-2D8h]
  __int64 v61; // [rsp+50h] [rbp-2D0h]
  _DWORD *v62; // [rsp+50h] [rbp-2D0h]
  __m128i v63; // [rsp+60h] [rbp-2C0h] BYREF
  void *s; // [rsp+70h] [rbp-2B0h] BYREF
  __int64 v65; // [rsp+78h] [rbp-2A8h]
  _BYTE v66[4]; // [rsp+80h] [rbp-2A0h] BYREF
  char v67; // [rsp+84h] [rbp-29Ch] BYREF
  __m128i src; // [rsp+A0h] [rbp-280h] BYREF
  _BYTE v69[80]; // [rsp+B0h] [rbp-270h] BYREF
  __m128i *v70; // [rsp+100h] [rbp-220h] BYREF
  __int64 v71; // [rsp+108h] [rbp-218h]
  _BYTE v72[528]; // [rsp+110h] [rbp-210h] BYREF

  v70 = *(__m128i **)(a2 + 112);
  if ( (unsigned __int8)sub_1560490(&v70, 11, 0) || (m128i_i64 = 18, (v54 = sub_1560180(a2 + 112, 18)) != 0) )
  {
    sub_1843A50((_QWORD *)a1, a2);
    return;
  }
  v5 = **(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL);
  v6 = *(_BYTE *)(v5 + 8);
  if ( v6 )
  {
    if ( v6 == 13 )
    {
      v59 = *(_DWORD *)(v5 + 12);
      goto LABEL_6;
    }
    if ( v6 == 14 )
    {
      v59 = *(_DWORD *)(v5 + 32);
LABEL_6:
      s = v66;
      v65 = 0x500000000LL;
      v7 = 6LL * v59;
      v52 = v59;
      n = 4LL * v59;
      if ( v59 <= 5uLL )
      {
        v8 = v66;
        v9 = &v66[4 * v59];
      }
      else
      {
        m128i_i64 = (__int64)v66;
        sub_16CD150((__int64)&s, v66, v59, 4, v3, v4);
        v8 = (char *)s;
        v9 = (char *)s + n;
      }
      goto LABEL_8;
    }
    v7 = 6;
    n = 4;
    v9 = &v67;
    v52 = 1;
    v59 = 1;
  }
  else
  {
    v7 = 0;
    n = 0;
    v9 = v66;
    v52 = 0;
    v59 = 0;
  }
  HIDWORD(v65) = 5;
  s = v66;
  v8 = v66;
LABEL_8:
  for ( LODWORD(v65) = v59; v8 != v9; v8 += 4 )
    *(_DWORD *)v8 = 1;
  src.m128i_i64[0] = (__int64)v69;
  src.m128i_i64[1] = 0x500000000LL;
  v70 = (__m128i *)v72;
  v71 = 0x500000000LL;
  if ( v52 > 5 )
  {
    m128i_i64 = v52;
    sub_1843B50((__int64)&v70, v52);
    v10 = v70;
  }
  else
  {
    v10 = (__m128i *)v72;
  }
  p_src = &src;
  v12 = &v10[v7];
  for ( LODWORD(v71) = v59; v12 != v10; v10 += 6 )
  {
    while ( 1 )
    {
      if ( v10 )
      {
        v13 = v10 + 1;
        v10->m128i_i32[2] = 0;
        v10->m128i_i64[0] = (__int64)v10[1].m128i_i64;
        v10->m128i_i32[3] = 5;
        v14 = src.m128i_i32[2];
        if ( v10 != &src )
        {
          if ( src.m128i_i32[2] )
            break;
        }
      }
      v10 += 6;
      if ( v12 == v10 )
        goto LABEL_21;
    }
    v15 = 16LL * src.m128i_u32[2];
    if ( src.m128i_i32[2] <= 5u
      || (m128i_i64 = (__int64)v10[1].m128i_i64,
          sub_16CD150((__int64)v10, &v10[1], src.m128i_u32[2], 16, v15, (int)p_src),
          v13 = (__m128i *)v10->m128i_i64[0],
          (v15 = 16LL * src.m128i_u32[2]) != 0) )
    {
      m128i_i64 = src.m128i_i64[0];
      memcpy(v13, (const void *)src.m128i_i64[0], v15);
    }
    v10->m128i_i32[2] = v14;
  }
LABEL_21:
  if ( (_BYTE *)src.m128i_i64[0] != v69 )
    _libc_free(src.m128i_u64[0]);
  v16 = *(_QWORD *)(a2 + 80);
  if ( v16 == a2 + 72 )
  {
    v53 = 0;
  }
  else
  {
    v17 = 0;
    do
    {
      v18 = v16 - 24;
      if ( !v16 )
        v18 = 0;
      v19 = sub_157EBA0(v18);
      if ( *(_BYTE *)(v19 + 16) == 25 )
      {
        v20 = *(_DWORD *)(v19 + 20) & 0xFFFFFFF;
        if ( v20 )
        {
          m128i_i64 = 4LL * v20;
          if ( **(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL) != **(_QWORD **)(v19 - 24LL * v20) )
            goto LABEL_40;
        }
      }
      v21 = sub_157EBE0(v18);
      v16 = *(_QWORD *)(v16 + 8);
      if ( v21 )
        v17 = 1;
    }
    while ( a2 + 72 != v16 );
    v53 = v17;
  }
  if ( (*(_BYTE *)(a2 + 32) & 0xFu) - 7 > 1 && (!*(_BYTE *)(a1 + 144) || (*(_BYTE *)(a2 + 33) & 0x20) != 0) )
  {
LABEL_40:
    sub_1843A50((_QWORD *)a1, a2);
    v25 = (unsigned __int64 *)v70;
    v26 = (unsigned __int64 *)&v70[6 * (unsigned int)v71];
    if ( v70 != (__m128i *)v26 )
    {
      do
      {
        v26 -= 12;
        if ( (unsigned __int64 *)*v26 != v26 + 2 )
          _libc_free(*v26);
      }
      while ( v25 != v26 );
      goto LABEL_44;
    }
    goto LABEL_45;
  }
  v56 = *(_QWORD *)(a2 + 8);
  if ( !v56 )
    goto LABEL_77;
  v58 = 0;
  v22 = 4LL * (v59 - 1) + 4;
  do
  {
    v23 = (unsigned __int64)sub_1648700(v56);
    v24 = *(_BYTE *)(v23 + 16);
    if ( v24 <= 0x17u )
      goto LABEL_40;
    if ( v24 == 78 )
    {
      v27 = v23 | 4;
    }
    else
    {
      if ( v24 != 29 )
        goto LABEL_40;
      v27 = v23 & 0xFFFFFFFFFFFFFFFBLL;
    }
    v28 = v27 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v27 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_40;
    v29 = (v27 >> 2) & 1;
    if ( v29 )
    {
      if ( v56 != v28 - 24 )
        goto LABEL_40;
      v30 = v54;
      if ( (*(_WORD *)(v28 + 18) & 3) == 2 )
        v30 = v29;
      v54 = v30;
    }
    else if ( v56 != v28 - 72 )
    {
      goto LABEL_40;
    }
    if ( v58 == v59 )
      goto LABEL_76;
    v31 = *(_QWORD *)(v28 + 8);
    if ( !v31 )
      goto LABEL_76;
    while ( 1 )
    {
      v32 = sub_1648700(v31);
      if ( *((_BYTE *)v32 + 16) != 86 )
        break;
      v49 = *(unsigned int *)v32[7];
      if ( *((_DWORD *)s + v49) )
      {
        m128i_i64 = (__int64)v32;
        v62 = (char *)s + 4 * v49;
        *v62 = sub_1843420((_QWORD *)a1, (__int64)v32, (__int64)v70[6 * v49].m128i_i64);
        v58 += *((_DWORD *)s + v49) == 0;
      }
LABEL_75:
      v31 = *(_QWORD *)(v31 + 8);
      if ( !v31 )
        goto LABEL_76;
    }
    m128i_i64 = v31;
    src.m128i_i64[0] = (__int64)v69;
    src.m128i_i64[1] = 0x500000000LL;
    if ( (unsigned int)sub_18430C0((_QWORD *)a1, v31, (__int64)&src, 0xFFFFFFFF) )
    {
      v35 = 0;
      if ( v59 )
      {
        do
        {
          while ( !*(_DWORD *)((char *)s + v35) )
          {
            v35 += 4;
            if ( v22 == v35 )
              goto LABEL_73;
          }
          v36 = src.m128i_i32[2];
          v37 = (const void *)src.m128i_i64[0];
          v38 = &v70[(unsigned __int64)(24 * v35) / 0x10];
          v39 = v38->m128i_u32[2];
          v40 = 16LL * src.m128i_u32[2];
          if ( src.m128i_u32[2] > (unsigned __int64)v38->m128i_u32[3] - v39 )
          {
            m128i_i64 = (__int64)v38[1].m128i_i64;
            v60 = src.m128i_i64[0];
            v61 = 16LL * src.m128i_u32[2];
            sub_16CD150(
              (__int64)v70[(unsigned __int64)(24 * v35) / 0x10].m128i_i64,
              &v38[1],
              src.m128i_u32[2] + v39,
              16,
              v40,
              src.m128i_i32[0]);
            v39 = v38->m128i_u32[2];
            v37 = (const void *)v60;
            v40 = v61;
          }
          if ( v40 )
          {
            m128i_i64 = (__int64)v37;
            memcpy((void *)(v38->m128i_i64[0] + 16 * v39), v37, v40);
            LODWORD(v39) = v38->m128i_i32[2];
          }
          v35 += 4;
          v38->m128i_i32[2] = v36 + v39;
        }
        while ( v22 != v35 );
      }
LABEL_73:
      if ( (_BYTE *)src.m128i_i64[0] != v69 )
        _libc_free(src.m128i_u64[0]);
      goto LABEL_75;
    }
    LODWORD(v65) = 0;
    if ( HIDWORD(v65) < v52 )
    {
      m128i_i64 = (__int64)v66;
      sub_16CD150((__int64)&s, v66, v52, 4, v33, v34);
    }
    LODWORD(v65) = v59;
    if ( n )
    {
      m128i_i64 = 0;
      memset(s, 0, n);
    }
    if ( (_BYTE *)src.m128i_i64[0] != v69 )
      _libc_free(src.m128i_u64[0]);
    v58 = v59;
LABEL_76:
    v56 = *(_QWORD *)(v56 + 8);
  }
  while ( v56 );
LABEL_77:
  v41 = 0;
  if ( v59 )
  {
    do
    {
      m128i_i64 = (__int64)&src;
      v42 = (__int64)v70[6 * v41].m128i_i64;
      v43 = *((_DWORD *)s + v41);
      src.m128i_i32[2] = v41++;
      src.m128i_i64[0] = a2;
      src.m128i_i8[12] = 0;
      sub_18437D0((_QWORD *)a1, &src, v43, v42);
    }
    while ( v59 != v41 );
  }
  src.m128i_i64[0] = (__int64)v69;
  src.m128i_i64[1] = 0x500000000LL;
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    sub_15E08E0(a2, m128i_i64);
    v44 = *(_QWORD *)(a2 + 88);
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
      sub_15E08E0(a2, m128i_i64);
    v45 = *(_QWORD *)(a2 + 88) + 40LL * *(_QWORD *)(a2 + 96);
    if ( v44 != v45 )
    {
LABEL_81:
      v46 = 0;
      v47 = v44;
      do
      {
        v48 = 0;
        if ( !(*(_DWORD *)(*(_QWORD *)(a2 + 24) + 8LL) >> 8) )
        {
          if ( v54 || v53 )
            v48 = 0;
          else
            v48 = sub_1843420((_QWORD *)a1, v47, (__int64)&src);
        }
        v63.m128i_i32[2] = v46;
        v63.m128i_i64[0] = a2;
        v47 += 40;
        ++v46;
        v63.m128i_i8[12] = 1;
        sub_18437D0((_QWORD *)a1, &v63, v48, (__int64)&src);
        src.m128i_i32[2] = 0;
      }
      while ( v47 != v45 );
    }
    if ( (_BYTE *)src.m128i_i64[0] != v69 )
      _libc_free(src.m128i_u64[0]);
  }
  else
  {
    v44 = *(_QWORD *)(a2 + 88);
    v45 = v44 + 40LL * *(_QWORD *)(a2 + 96);
    if ( v45 != v44 )
      goto LABEL_81;
  }
  v50 = (unsigned __int64 *)v70;
  v26 = (unsigned __int64 *)&v70[6 * (unsigned int)v71];
  if ( v70 != (__m128i *)v26 )
  {
    do
    {
      v26 -= 12;
      if ( (unsigned __int64 *)*v26 != v26 + 2 )
        _libc_free(*v26);
    }
    while ( v50 != v26 );
LABEL_44:
    v26 = (unsigned __int64 *)v70;
  }
LABEL_45:
  if ( v26 != (unsigned __int64 *)v72 )
    _libc_free((unsigned __int64)v26);
  if ( s != v66 )
    _libc_free((unsigned __int64)s);
}

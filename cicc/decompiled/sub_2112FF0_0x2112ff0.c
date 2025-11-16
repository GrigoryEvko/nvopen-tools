// Function: sub_2112FF0
// Address: 0x2112ff0
//
void __fastcall sub_2112FF0(__int64 a1, __int64 a2, __int64 j, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // r14d
  __int64 v8; // r13
  unsigned int i; // ebx
  __int64 v10; // rdi
  __int64 *v11; // rax
  __int64 *v12; // r14
  int v13; // eax
  __int64 v14; // rdi
  unsigned int v15; // edx
  unsigned __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rbx
  __int64 v19; // r13
  unsigned __int64 v20; // rdi
  int v21; // ecx
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // rbx
  __int64 v25; // r13
  __int64 v26; // rdi
  unsigned int k; // r13d
  unsigned int v28; // esi
  unsigned int v29; // ecx
  unsigned int v30; // edx
  unsigned int v31; // eax
  unsigned __int64 v32; // rdi
  __m128i *v33; // r15
  int v34; // r11d
  __int64 *v35; // r10
  int v36; // ecx
  int v37; // eax
  int v38; // r11d
  __int64 *v39; // r10
  unsigned int v40; // edx
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 m; // r12
  __int64 v44; // rdi
  int v45; // r11d
  __m128i v46; // [rsp+0h] [rbp-50h] BYREF
  __int32 v47; // [rsp+10h] [rbp-40h]

  v7 = *(_DWORD *)(a1 + 128);
  if ( !v7 )
  {
    v16 = *(unsigned int *)(a1 + 176);
    if ( *(_DWORD *)(a1 + 176) )
    {
      j = 0;
      goto LABEL_14;
    }
    goto LABEL_17;
  }
  v8 = a1 + 136;
  for ( i = 0; i < v7; ++i )
  {
    a2 = *(unsigned int *)(a1 + 160);
    v12 = (__int64 *)(*(_QWORD *)(a1 + 112) + 8LL * i);
    if ( (_DWORD)a2 )
    {
      a4 = *v12;
      LODWORD(a5) = a2 - 1;
      v10 = *(_QWORD *)(a1 + 144);
      j = ((_DWORD)a2 - 1) & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4));
      v11 = (__int64 *)(v10 + 16 * j);
      a6 = *v11;
      if ( *v12 == *v11 )
        goto LABEL_4;
      v34 = 1;
      v35 = 0;
      while ( a6 != -8 )
      {
        if ( !v35 && a6 == -16 )
          v35 = v11;
        j = (unsigned int)a5 & (v34 + (_DWORD)j);
        v11 = (__int64 *)(v10 + 16LL * (unsigned int)j);
        a6 = *v11;
        if ( a4 == *v11 )
          goto LABEL_4;
        ++v34;
      }
      v36 = *(_DWORD *)(a1 + 152);
      if ( v35 )
        v11 = v35;
      ++*(_QWORD *)(a1 + 136);
      LODWORD(a4) = v36 + 1;
      if ( 4 * (int)a4 < (unsigned int)(3 * a2) )
      {
        if ( (int)a2 - *(_DWORD *)(a1 + 156) - (int)a4 > (unsigned int)a2 >> 3 )
          goto LABEL_9;
        sub_1B36490(v8, a2);
        v37 = *(_DWORD *)(a1 + 160);
        if ( !v37 )
        {
LABEL_77:
          ++*(_DWORD *)(a1 + 152);
          BUG();
        }
        a2 = (unsigned int)(v37 - 1);
        v38 = 1;
        v39 = 0;
        a5 = *(_QWORD *)(a1 + 144);
        LODWORD(a4) = *(_DWORD *)(a1 + 152) + 1;
        v40 = a2 & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4));
        v11 = (__int64 *)(a5 + 16LL * v40);
        a6 = *v11;
        if ( *v11 == *v12 )
          goto LABEL_9;
        while ( a6 != -8 )
        {
          if ( !v39 && a6 == -16 )
            v39 = v11;
          v40 = a2 & (v38 + v40);
          v11 = (__int64 *)(a5 + 16LL * v40);
          a6 = *v11;
          if ( *v12 == *v11 )
            goto LABEL_9;
          ++v38;
        }
        goto LABEL_43;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 136);
    }
    sub_1B36490(v8, 2 * a2);
    v13 = *(_DWORD *)(a1 + 160);
    if ( !v13 )
      goto LABEL_77;
    a2 = *v12;
    LODWORD(a5) = v13 - 1;
    v14 = *(_QWORD *)(a1 + 144);
    LODWORD(a4) = *(_DWORD *)(a1 + 152) + 1;
    v15 = (v13 - 1) & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4));
    v11 = (__int64 *)(v14 + 16LL * v15);
    a6 = *v11;
    if ( *v12 == *v11 )
      goto LABEL_9;
    v45 = 1;
    v39 = 0;
    while ( a6 != -8 )
    {
      if ( !v39 && a6 == -16 )
        v39 = v11;
      v15 = a5 & (v45 + v15);
      v11 = (__int64 *)(v14 + 16LL * v15);
      a6 = *v11;
      if ( a2 == *v11 )
        goto LABEL_9;
      ++v45;
    }
LABEL_43:
    if ( v39 )
      v11 = v39;
LABEL_9:
    *(_DWORD *)(a1 + 152) = a4;
    if ( *v11 != -8 )
      --*(_DWORD *)(a1 + 156);
    j = *v12;
    *((_DWORD *)v11 + 2) = 0;
    *v11 = j;
LABEL_4:
    *((_DWORD *)v11 + 2) = i;
    v7 = *(_DWORD *)(a1 + 128);
  }
  v16 = *(unsigned int *)(a1 + 176);
  if ( v7 >= v16 )
  {
    if ( v7 > v16 )
    {
      if ( v7 > (unsigned __int64)*(unsigned int *)(a1 + 180) )
      {
        a2 = v7;
        sub_210E3D0(a1 + 168, v7);
        v16 = *(unsigned int *)(a1 + 176);
      }
      v41 = *(_QWORD *)(a1 + 168);
      LODWORD(a4) = 3 * v7;
      v42 = v41 + 24 * v16;
      for ( j = v41 + 24LL * v7; j != v42; v42 += 24 )
      {
        if ( v42 )
        {
          *(_QWORD *)v42 = 0;
          *(_QWORD *)(v42 + 8) = 0;
          *(_DWORD *)(v42 + 16) = 0;
        }
      }
      goto LABEL_16;
    }
  }
  else
  {
    j = 24LL * v7;
LABEL_14:
    v17 = *(_QWORD *)(a1 + 168);
    v18 = v17 + 24 * v16;
    v19 = j + v17;
    while ( v19 != v18 )
    {
      v20 = *(_QWORD *)(v18 - 24);
      v18 -= 24;
      _libc_free(v20);
    }
LABEL_16:
    *(_DWORD *)(a1 + 176) = v7;
  }
LABEL_17:
  sub_2111830(a1, a2, j, a4, a5, a6);
  v24 = *(_QWORD *)(a1 + 168);
  if ( byte_4FCFBE0 )
  {
    v25 = v24 + 24LL * *(unsigned int *)(a1 + 176);
    while ( v25 != v24 )
    {
      v26 = v24;
      v24 += 24;
      sub_13A49F0(v26, *(_DWORD *)(a1 + 40), 0, v21, v22, v23);
    }
    for ( k = 0; *(_DWORD *)(a1 + 128) > k; ++k )
    {
      while ( (*(_QWORD *)(*(_QWORD *)(a1 + 376) + 8LL * (k >> 6)) & (1LL << k)) != 0 )
      {
        if ( *(_DWORD *)(a1 + 128) <= ++k )
          goto LABEL_33;
      }
      v28 = *(_DWORD *)(a1 + 40);
      v46 = 0u;
      v47 = 0;
      sub_13A49F0((__int64)&v46, v28, 0, k, v22, v23);
      v29 = *(_DWORD *)(a1 + 40);
      if ( v29 )
      {
        if ( v29 >> 6 )
        {
          *(_QWORD *)v46.m128i_i64[0] = -1;
          v30 = 64;
          do
          {
            *(_QWORD *)(v46.m128i_i64[0] + 8LL * ((v30 - 64) >> 6)) = -1;
            v31 = v30;
            v30 += 64;
          }
          while ( v29 >= v30 );
          if ( v29 > v31 )
            *(_QWORD *)(v46.m128i_i64[0] + 8LL * (v31 >> 6)) |= (1LL << v29) - 1;
        }
        else
        {
          *(_QWORD *)v46.m128i_i64[0] |= (1LL << v29) - 1;
        }
      }
      v32 = v46.m128i_i64[0];
      v33 = (__m128i *)(*(_QWORD *)(a1 + 168) + 24LL * k);
      if ( v33 != &v46 )
      {
        _libc_free(v33->m128i_i64[0]);
        v32 = 0;
        *v33 = _mm_loadu_si128(&v46);
        v33[1].m128i_i32[0] = v47;
      }
      _libc_free(v32);
    }
LABEL_33:
    sub_210F050(a1);
    sub_21101B0(a1);
  }
  else
  {
    for ( m = v24 + 24LL * *(unsigned int *)(a1 + 176); m != v24; **(_QWORD **)(v24 - 24) |= 1uLL )
    {
      v44 = v24;
      v24 += 24;
      sub_13A49F0(v44, 1u, 0, v21, v22, v23);
    }
  }
}

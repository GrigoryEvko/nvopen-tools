// Function: sub_F62A00
// Address: 0xf62a00
//
__int64 __fastcall sub_F62A00(__int64 a1, const __m128i *a2)
{
  __int64 v4; // r13
  __int64 v5; // r14
  unsigned int v6; // esi
  __int64 v7; // rcx
  int v8; // r11d
  unsigned int v9; // edi
  unsigned __int64 v10; // rax
  __int64 *v11; // rdx
  unsigned int i; // r9d
  __int64 *v13; // rdi
  __int64 v14; // r8
  unsigned int v15; // r9d
  int v17; // ecx
  int v18; // edi
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rcx
  int v21; // eax
  __int64 v22; // rcx
  __m128i *v23; // rdx
  __m128i v24; // xmm0
  __int64 v25; // rax
  __m128i v26; // xmm1
  unsigned __int64 v27; // r8
  unsigned __int64 v28; // rsi
  const __m128i *v29; // rax
  __m128i *v30; // rdx
  int v31; // edx
  __int64 v32; // rcx
  int v33; // r9d
  __int64 *v34; // r8
  int v35; // edi
  unsigned int j; // eax
  __int64 v37; // rsi
  unsigned int v38; // eax
  int v39; // edx
  int v40; // ecx
  __int64 v41; // rdi
  int v42; // r9d
  unsigned int k; // eax
  __int64 v44; // rsi
  unsigned int v45; // eax
  __int64 v46; // rdi
  __int64 v47; // r9
  __int8 *v48; // r12
  int v49; // [rsp+8h] [rbp-48h]
  __m128i v50; // [rsp+10h] [rbp-40h] BYREF
  __int64 v51; // [rsp+20h] [rbp-30h]

  v4 = a2->m128i_i64[0];
  v5 = a2->m128i_i64[1];
  v6 = *(_DWORD *)(a1 + 24);
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_29;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v9 = (unsigned int)v5 >> 9;
  v10 = ((0xBF58476D1CE4E5B9LL
        * (v9 ^ ((unsigned int)v5 >> 4) | ((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32))) >> 31)
      ^ (0xBF58476D1CE4E5B9LL
       * (v9 ^ ((unsigned int)v5 >> 4) | ((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32)));
  v11 = 0;
  for ( i = (((0xBF58476D1CE4E5B9LL
             * (v9 ^ ((unsigned int)v5 >> 4)
              | ((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32))) >> 31)
           ^ (484763065 * (v9 ^ ((unsigned int)v5 >> 4))))
          & (v6 - 1); ; i = (v6 - 1) & v15 )
  {
    v13 = (__int64 *)(v7 + 24LL * i);
    v14 = *v13;
    if ( v4 == *v13 && v13[1] == v5 )
      return *(_QWORD *)(a1 + 32) + 24LL * *((unsigned int *)v13 + 4);
    if ( v14 == -4096 )
      break;
    if ( v14 == -8192 && v13[1] == -8192 && !v11 )
      v11 = (__int64 *)(v7 + 24LL * i);
LABEL_9:
    v15 = v8 + i;
    ++v8;
  }
  if ( v13[1] != -4096 )
    goto LABEL_9;
  v17 = *(_DWORD *)(a1 + 16);
  if ( !v11 )
    v11 = v13;
  ++*(_QWORD *)a1;
  v18 = v17 + 1;
  if ( 4 * (v17 + 1) >= 3 * v6 )
  {
LABEL_29:
    sub_F62730(a1, 2 * v6);
    v31 = *(_DWORD *)(a1 + 24);
    if ( v31 )
    {
      v33 = 1;
      v34 = 0;
      v35 = v31 - 1;
      for ( j = (v31 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)
                  | ((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)))); ; j = v35 & v38 )
      {
        v32 = *(_QWORD *)(a1 + 8);
        v11 = (__int64 *)(v32 + 24LL * j);
        v37 = *v11;
        if ( v4 == *v11 && v11[1] == v5 )
          break;
        if ( v37 == -4096 )
        {
          if ( v11[1] == -4096 )
          {
LABEL_56:
            if ( v34 )
              v11 = v34;
            v18 = *(_DWORD *)(a1 + 16) + 1;
            goto LABEL_17;
          }
        }
        else if ( v37 == -8192 && v11[1] == -8192 && !v34 )
        {
          v34 = (__int64 *)(v32 + 24LL * j);
        }
        v38 = v33 + j;
        ++v33;
      }
      goto LABEL_48;
    }
LABEL_61:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v6 - *(_DWORD *)(a1 + 20) - v18 <= v6 >> 3 )
  {
    v49 = v10;
    sub_F62730(a1, v6);
    v39 = *(_DWORD *)(a1 + 24);
    if ( v39 )
    {
      v40 = v39 - 1;
      v34 = 0;
      v42 = 1;
      for ( k = (v39 - 1) & v49; ; k = v40 & v45 )
      {
        v41 = *(_QWORD *)(a1 + 8);
        v11 = (__int64 *)(v41 + 24LL * k);
        v44 = *v11;
        if ( v4 == *v11 && v11[1] == v5 )
          break;
        if ( v44 == -4096 )
        {
          if ( v11[1] == -4096 )
            goto LABEL_56;
        }
        else if ( v44 == -8192 && v11[1] == -8192 && !v34 )
        {
          v34 = (__int64 *)(v41 + 24LL * k);
        }
        v45 = v42 + k;
        ++v42;
      }
LABEL_48:
      v18 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_17;
    }
    goto LABEL_61;
  }
LABEL_17:
  *(_DWORD *)(a1 + 16) = v18;
  if ( *v11 != -4096 || v11[1] != -4096 )
    --*(_DWORD *)(a1 + 20);
  *((_DWORD *)v11 + 4) = 0;
  *v11 = v4;
  v11[1] = v5;
  *((_DWORD *)v11 + 4) = *(_DWORD *)(a1 + 40);
  v19 = *(unsigned int *)(a1 + 40);
  v20 = *(unsigned int *)(a1 + 44);
  v21 = *(_DWORD *)(a1 + 40);
  if ( v19 >= v20 )
  {
    v26 = _mm_loadu_si128(a2);
    v27 = v19 + 1;
    v28 = *(_QWORD *)(a1 + 32);
    v29 = &v50;
    v51 = 0;
    v50 = v26;
    if ( v20 < v19 + 1 )
    {
      v46 = a1 + 32;
      v47 = a1 + 48;
      if ( v28 > (unsigned __int64)&v50 || (unsigned __int64)&v50 >= v28 + 24 * v19 )
      {
        sub_C8D5F0(v46, (const void *)(a1 + 48), v27, 0x18u, v27, v47);
        v28 = *(_QWORD *)(a1 + 32);
        v19 = *(unsigned int *)(a1 + 40);
        v29 = &v50;
      }
      else
      {
        v48 = &v50.m128i_i8[-v28];
        sub_C8D5F0(v46, (const void *)(a1 + 48), v27, 0x18u, v27, v47);
        v28 = *(_QWORD *)(a1 + 32);
        v19 = *(unsigned int *)(a1 + 40);
        v29 = (const __m128i *)&v48[v28];
      }
    }
    v30 = (__m128i *)(v28 + 24 * v19);
    *v30 = _mm_loadu_si128(v29);
    v30[1].m128i_i64[0] = v29[1].m128i_i64[0];
    v22 = *(_QWORD *)(a1 + 32);
    v25 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
    *(_DWORD *)(a1 + 40) = v25;
  }
  else
  {
    v22 = *(_QWORD *)(a1 + 32);
    v23 = (__m128i *)(v22 + 24 * v19);
    if ( v23 )
    {
      v24 = _mm_loadu_si128(a2);
      v23[1].m128i_i64[0] = 0;
      *v23 = v24;
      v21 = *(_DWORD *)(a1 + 40);
      v22 = *(_QWORD *)(a1 + 32);
    }
    v25 = (unsigned int)(v21 + 1);
    *(_DWORD *)(a1 + 40) = v25;
  }
  return v22 + 24 * v25 - 24;
}

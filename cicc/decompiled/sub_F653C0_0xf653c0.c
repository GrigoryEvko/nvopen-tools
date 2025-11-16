// Function: sub_F653C0
// Address: 0xf653c0
//
__int64 __fastcall sub_F653C0(__int64 a1, const __m128i *a2, __int64 *a3)
{
  __int64 v6; // r13
  __int64 v7; // r15
  unsigned int v8; // esi
  __int64 v9; // rcx
  int v10; // r10d
  unsigned int v11; // edi
  unsigned __int64 v12; // rax
  __int64 *v13; // rdx
  unsigned int i; // r8d
  __int64 *v15; // rdi
  __int64 v16; // r11
  unsigned int v17; // r8d
  int v19; // ecx
  int v20; // edi
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  int v23; // eax
  __int64 v24; // rcx
  __m128i *v25; // rdx
  __m128i v26; // xmm0
  __int64 v27; // rax
  __m128i v28; // xmm1
  unsigned __int64 v29; // r8
  unsigned __int64 v30; // rsi
  const __m128i *v31; // rax
  __m128i *v32; // rdx
  int v33; // edx
  __int64 v34; // rcx
  int v35; // r9d
  __int64 *v36; // r8
  int v37; // edi
  unsigned int j; // eax
  __int64 v39; // rsi
  unsigned int v40; // eax
  int v41; // edx
  int v42; // ecx
  __int64 v43; // rdi
  int v44; // r9d
  unsigned int k; // eax
  __int64 v46; // rsi
  unsigned int v47; // eax
  __int64 v48; // rdi
  __int64 v49; // r9
  __int8 *v50; // r12
  int v51; // [rsp+8h] [rbp-58h]
  __m128i v52; // [rsp+10h] [rbp-50h] BYREF
  __int64 v53; // [rsp+20h] [rbp-40h]

  v6 = a2->m128i_i64[0];
  v7 = a2->m128i_i64[1];
  v8 = *(_DWORD *)(a1 + 24);
  if ( !v8 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_29;
  }
  v9 = *(_QWORD *)(a1 + 8);
  v10 = 1;
  v11 = (unsigned int)v7 >> 9;
  v12 = ((0xBF58476D1CE4E5B9LL
        * (v11 ^ ((unsigned int)v7 >> 4) | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))) >> 31)
      ^ (0xBF58476D1CE4E5B9LL
       * (v11 ^ ((unsigned int)v7 >> 4) | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32)));
  v13 = 0;
  for ( i = (((0xBF58476D1CE4E5B9LL
             * (v11 ^ ((unsigned int)v7 >> 4)
              | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))) >> 31)
           ^ (484763065 * (v11 ^ ((unsigned int)v7 >> 4))))
          & (v8 - 1); ; i = (v8 - 1) & v17 )
  {
    v15 = (__int64 *)(v9 + 24LL * i);
    v16 = *v15;
    if ( v6 == *v15 && v15[1] == v7 )
      return *(_QWORD *)(a1 + 32) + 24LL * *((unsigned int *)v15 + 4);
    if ( v16 == -4096 )
      break;
    if ( v16 == -8192 && v15[1] == -8192 && !v13 )
      v13 = (__int64 *)(v9 + 24LL * i);
LABEL_9:
    v17 = v10 + i;
    ++v10;
  }
  if ( v15[1] != -4096 )
    goto LABEL_9;
  v19 = *(_DWORD *)(a1 + 16);
  if ( !v13 )
    v13 = v15;
  ++*(_QWORD *)a1;
  v20 = v19 + 1;
  if ( 4 * (v19 + 1) >= 3 * v8 )
  {
LABEL_29:
    sub_F650F0(a1, 2 * v8);
    v33 = *(_DWORD *)(a1 + 24);
    if ( v33 )
    {
      v35 = 1;
      v36 = 0;
      v37 = v33 - 1;
      for ( j = (v33 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)
                  | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)))); ; j = v37 & v40 )
      {
        v34 = *(_QWORD *)(a1 + 8);
        v13 = (__int64 *)(v34 + 24LL * j);
        v39 = *v13;
        if ( v6 == *v13 && v13[1] == v7 )
          break;
        if ( v39 == -4096 )
        {
          if ( v13[1] == -4096 )
          {
LABEL_56:
            if ( v36 )
              v13 = v36;
            v20 = *(_DWORD *)(a1 + 16) + 1;
            goto LABEL_17;
          }
        }
        else if ( v39 == -8192 && v13[1] == -8192 && !v36 )
        {
          v36 = (__int64 *)(v34 + 24LL * j);
        }
        v40 = v35 + j;
        ++v35;
      }
      goto LABEL_48;
    }
LABEL_61:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v8 - *(_DWORD *)(a1 + 20) - v20 <= v8 >> 3 )
  {
    v51 = v12;
    sub_F650F0(a1, v8);
    v41 = *(_DWORD *)(a1 + 24);
    if ( v41 )
    {
      v42 = v41 - 1;
      v36 = 0;
      v44 = 1;
      for ( k = (v41 - 1) & v51; ; k = v42 & v47 )
      {
        v43 = *(_QWORD *)(a1 + 8);
        v13 = (__int64 *)(v43 + 24LL * k);
        v46 = *v13;
        if ( v6 == *v13 && v13[1] == v7 )
          break;
        if ( v46 == -4096 )
        {
          if ( v13[1] == -4096 )
            goto LABEL_56;
        }
        else if ( v46 == -8192 && v13[1] == -8192 && !v36 )
        {
          v36 = (__int64 *)(v43 + 24LL * k);
        }
        v47 = v44 + k;
        ++v44;
      }
LABEL_48:
      v20 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_17;
    }
    goto LABEL_61;
  }
LABEL_17:
  *(_DWORD *)(a1 + 16) = v20;
  if ( *v13 != -4096 || v13[1] != -4096 )
    --*(_DWORD *)(a1 + 20);
  *((_DWORD *)v13 + 4) = 0;
  *v13 = v6;
  v13[1] = v7;
  *((_DWORD *)v13 + 4) = *(_DWORD *)(a1 + 40);
  v21 = *(unsigned int *)(a1 + 40);
  v22 = *(unsigned int *)(a1 + 44);
  v23 = *(_DWORD *)(a1 + 40);
  if ( v21 >= v22 )
  {
    v28 = _mm_loadu_si128(a2);
    v29 = v21 + 1;
    v30 = *(_QWORD *)(a1 + 32);
    v53 = *a3;
    v31 = &v52;
    v52 = v28;
    if ( v22 < v21 + 1 )
    {
      v48 = a1 + 32;
      v49 = a1 + 48;
      if ( v30 > (unsigned __int64)&v52 || (unsigned __int64)&v52 >= v30 + 24 * v21 )
      {
        sub_C8D5F0(v48, (const void *)(a1 + 48), v29, 0x18u, v29, v49);
        v30 = *(_QWORD *)(a1 + 32);
        v21 = *(unsigned int *)(a1 + 40);
        v31 = &v52;
      }
      else
      {
        v50 = &v52.m128i_i8[-v30];
        sub_C8D5F0(v48, (const void *)(a1 + 48), v29, 0x18u, v29, v49);
        v30 = *(_QWORD *)(a1 + 32);
        v21 = *(unsigned int *)(a1 + 40);
        v31 = (const __m128i *)&v50[v30];
      }
    }
    v32 = (__m128i *)(v30 + 24 * v21);
    *v32 = _mm_loadu_si128(v31);
    v32[1].m128i_i64[0] = v31[1].m128i_i64[0];
    v24 = *(_QWORD *)(a1 + 32);
    v27 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
    *(_DWORD *)(a1 + 40) = v27;
  }
  else
  {
    v24 = *(_QWORD *)(a1 + 32);
    v25 = (__m128i *)(v24 + 24 * v21);
    if ( v25 )
    {
      v26 = _mm_loadu_si128(a2);
      v25[1].m128i_i64[0] = *a3;
      *v25 = v26;
      v23 = *(_DWORD *)(a1 + 40);
      v24 = *(_QWORD *)(a1 + 32);
    }
    v27 = (unsigned int)(v23 + 1);
    *(_DWORD *)(a1 + 40) = v27;
  }
  return v24 + 24 * v27 - 24;
}

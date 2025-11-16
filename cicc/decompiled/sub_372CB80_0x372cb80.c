// Function: sub_372CB80
// Address: 0x372cb80
//
__int64 __fastcall sub_372CB80(__int64 a1, const __m128i *a2)
{
  __int64 v4; // r13
  __int64 v5; // r14
  unsigned int v6; // esi
  __int64 v7; // rcx
  __int64 v8; // r9
  int v9; // r10d
  __int64 *v10; // r15
  __int64 i; // r8
  __int64 *v12; // rdx
  __int64 v13; // rdi
  int v14; // r8d
  __int64 v15; // rax
  int v17; // ecx
  int v18; // ecx
  __int64 v19; // rcx
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rsi
  __int64 v22; // rdx
  __m128i *v23; // rsi
  __m128i *v24; // rdi
  __m128i v25; // xmm1
  _BYTE *v26; // rdi
  unsigned __int64 v27; // r14
  __int64 v28; // rdi
  int v29; // ecx
  int v30; // ecx
  __int64 v31; // rdx
  __int64 *v32; // rdi
  unsigned int j; // eax
  __int64 v34; // rsi
  int v35; // eax
  int v36; // edx
  int v37; // edx
  __int64 v38; // rsi
  unsigned int k; // eax
  __int64 v40; // rcx
  int v41; // eax
  __m128i v42; // [rsp+60h] [rbp-90h] BYREF
  _BYTE *v43; // [rsp+70h] [rbp-80h]
  __int64 v44; // [rsp+78h] [rbp-78h]
  _BYTE v45[112]; // [rsp+80h] [rbp-70h] BYREF

  v4 = a2->m128i_i64[0];
  v5 = a2->m128i_i64[1];
  v6 = *(_DWORD *)(a1 + 24);
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_35;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = v6 - 1;
  v9 = 1;
  v10 = 0;
  for ( i = ((unsigned int)((0xBF58476D1CE4E5B9LL
                           * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)
                            | ((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4))))
          & (v6 - 1); ; i = (unsigned int)v8 & v14 )
  {
    v12 = (__int64 *)(v7 + 24LL * (unsigned int)i);
    v13 = *v12;
    if ( v4 == *v12 && v12[1] == v5 )
    {
      v15 = *((unsigned int *)v12 + 4);
      return *(_QWORD *)(a1 + 32) + 96 * v15 + 16;
    }
    if ( v13 == -4096 )
      break;
    if ( v13 == -8192 && v12[1] == -8192 && !v10 )
      v10 = (__int64 *)(v7 + 24LL * (unsigned int)i);
LABEL_9:
    v14 = v9 + i;
    ++v9;
  }
  if ( v12[1] != -4096 )
    goto LABEL_9;
  v17 = *(_DWORD *)(a1 + 16);
  if ( !v10 )
    v10 = v12;
  ++*(_QWORD *)a1;
  v18 = v17 + 1;
  if ( 4 * v18 >= 3 * v6 )
  {
LABEL_35:
    sub_372C8B0(a1, 2 * v6);
    v29 = *(_DWORD *)(a1 + 24);
    if ( v29 )
    {
      v30 = v29 - 1;
      i = 1;
      v32 = 0;
      for ( j = v30
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)
                  | ((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)))); ; j = v30 & v35 )
      {
        v31 = *(_QWORD *)(a1 + 8);
        v10 = (__int64 *)(v31 + 24LL * j);
        v34 = *v10;
        if ( v4 == *v10 && v10[1] == v5 )
          break;
        if ( v34 == -4096 )
        {
          if ( v10[1] == -4096 )
          {
LABEL_59:
            if ( v32 )
              v10 = v32;
            v18 = *(_DWORD *)(a1 + 16) + 1;
            goto LABEL_18;
          }
        }
        else if ( v34 == -8192 && v10[1] == -8192 && !v32 )
        {
          v32 = (__int64 *)(v31 + 24LL * j);
        }
        v35 = i + j;
        i = (unsigned int)(i + 1);
      }
      goto LABEL_55;
    }
LABEL_64:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v6 - *(_DWORD *)(a1 + 20) - v18 <= v6 >> 3 )
  {
    sub_372C8B0(a1, v6);
    v36 = *(_DWORD *)(a1 + 24);
    if ( v36 )
    {
      v37 = v36 - 1;
      v32 = 0;
      i = 1;
      for ( k = v37
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)
                  | ((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)))); ; k = v37 & v41 )
      {
        v38 = *(_QWORD *)(a1 + 8);
        v10 = (__int64 *)(v38 + 24LL * k);
        v40 = *v10;
        if ( v4 == *v10 && v10[1] == v5 )
          break;
        if ( v40 == -4096 )
        {
          if ( v10[1] == -4096 )
            goto LABEL_59;
        }
        else if ( v40 == -8192 && v10[1] == -8192 && !v32 )
        {
          v32 = (__int64 *)(v38 + 24LL * k);
        }
        v41 = i + k;
        i = (unsigned int)(i + 1);
      }
LABEL_55:
      v18 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_18;
    }
    goto LABEL_64;
  }
LABEL_18:
  *(_DWORD *)(a1 + 16) = v18;
  if ( *v10 != -4096 || v10[1] != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v10 = v4;
  v10[1] = v5;
  *((_DWORD *)v10 + 4) = 0;
  v19 = *(unsigned int *)(a1 + 40);
  v20 = *(unsigned int *)(a1 + 44);
  v43 = v45;
  v21 = v19 + 1;
  v44 = 0x400000000LL;
  v15 = v19;
  v42 = _mm_loadu_si128(a2);
  if ( v19 + 1 > v20 )
  {
    v27 = *(_QWORD *)(a1 + 32);
    v28 = a1 + 32;
    if ( v27 > (unsigned __int64)&v42 || (unsigned __int64)&v42 >= v27 + 96 * v19 )
    {
      sub_372B930(v28, v21, v20, v19, i, v8);
      v19 = *(unsigned int *)(a1 + 40);
      v22 = *(_QWORD *)(a1 + 32);
      v23 = &v42;
      v15 = v19;
    }
    else
    {
      sub_372B930(v28, v21, v20, v19, i, v8);
      v22 = *(_QWORD *)(a1 + 32);
      v19 = *(unsigned int *)(a1 + 40);
      v23 = (__m128i *)((char *)&v42 + v22 - v27);
      v15 = v19;
    }
  }
  else
  {
    v22 = *(_QWORD *)(a1 + 32);
    v23 = &v42;
  }
  v24 = (__m128i *)(v22 + 96 * v19);
  if ( v24 )
  {
    v25 = _mm_loadu_si128(v23);
    v24[1].m128i_i64[0] = (__int64)v24[2].m128i_i64;
    v24[1].m128i_i64[1] = 0x400000000LL;
    *v24 = v25;
    if ( v23[1].m128i_i32[2] )
      sub_372A010((__int64)v24[1].m128i_i64, (char **)&v23[1], v22, v19, i, v8);
    v15 = *(unsigned int *)(a1 + 40);
  }
  v26 = v43;
  *(_DWORD *)(a1 + 40) = v15 + 1;
  if ( v26 != v45 )
  {
    _libc_free((unsigned __int64)v26);
    v15 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  }
  *((_DWORD *)v10 + 4) = v15;
  return *(_QWORD *)(a1 + 32) + 96 * v15 + 16;
}

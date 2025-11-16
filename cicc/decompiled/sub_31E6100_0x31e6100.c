// Function: sub_31E6100
// Address: 0x31e6100
//
__int64 __fastcall sub_31E6100(__int64 a1, int *a2)
{
  int v4; // r13d
  int v5; // r14d
  unsigned int v6; // esi
  __int64 v7; // rcx
  _DWORD *v8; // r15
  int v9; // r10d
  __int64 v10; // r9
  __int64 i; // r8
  _DWORD *v12; // rdi
  int v13; // r8d
  __int64 v14; // rax
  int v16; // edx
  int v17; // edx
  int v18; // edx
  __int64 v19; // rcx
  _DWORD *v20; // rdi
  unsigned int j; // eax
  int v22; // eax
  int v23; // ecx
  int v24; // ecx
  __int64 v25; // rax
  unsigned __int64 v26; // rcx
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 v29; // rcx
  const __m128i *v30; // rdx
  __m128i *v31; // rax
  unsigned __int64 v32; // r13
  __int64 v33; // rdi
  const void *v34; // rsi
  int v35; // edx
  int v36; // edx
  __int64 v37; // rsi
  unsigned int k; // eax
  int v39; // eax
  int v40; // ecx
  int v41; // esi
  _QWORD v42[10]; // [rsp+10h] [rbp-50h] BYREF

  v4 = *a2;
  v5 = a2[1];
  v6 = *(_DWORD *)(a1 + 24);
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_15;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 0;
  v9 = 1;
  v10 = v6 - 1;
  for ( i = ((unsigned int)((0xBF58476D1CE4E5B9LL
                           * ((unsigned int)(37 * v5) | ((unsigned __int64)(unsigned int)(37 * v4) << 32))) >> 31)
           ^ (756364221 * v5))
          & (v6 - 1); ; i = (unsigned int)v10 & v13 )
  {
    v12 = (_DWORD *)(v7 + 12LL * (unsigned int)i);
    if ( v4 == *v12 && v5 == v12[1] )
    {
      v14 = (unsigned int)v12[2];
      return *(_QWORD *)(a1 + 32) + 24 * v14 + 8;
    }
    if ( !*v12 )
      break;
LABEL_5:
    v13 = v9 + i;
    ++v9;
  }
  v16 = v12[1];
  if ( v16 != -1 )
  {
    if ( !v8 && v16 == -2 )
      v8 = (_DWORD *)(v7 + 12LL * (unsigned int)i);
    goto LABEL_5;
  }
  v23 = *(_DWORD *)(a1 + 16);
  if ( !v8 )
    v8 = v12;
  ++*(_QWORD *)a1;
  v24 = v23 + 1;
  if ( 4 * v24 >= 3 * v6 )
  {
LABEL_15:
    sub_31E5EB0(a1, 2 * v6);
    v17 = *(_DWORD *)(a1 + 24);
    if ( v17 )
    {
      v18 = v17 - 1;
      i = 1;
      v20 = 0;
      for ( j = v18
              & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v5) | ((unsigned __int64)(unsigned int)(37 * v4) << 32))) >> 31)
               ^ (756364221 * v5)); ; j = v18 & v22 )
      {
        v19 = *(_QWORD *)(a1 + 8);
        v8 = (_DWORD *)(v19 + 12LL * j);
        if ( v4 == *v8 && v5 == v8[1] )
          break;
        if ( !*v8 )
        {
          v41 = v8[1];
          if ( v41 == -1 )
          {
LABEL_54:
            if ( v20 )
              v8 = v20;
            v24 = *(_DWORD *)(a1 + 16) + 1;
            goto LABEL_24;
          }
          if ( v41 == -2 && !v20 )
            v20 = (_DWORD *)(v19 + 12LL * j);
        }
        v22 = i + j;
        i = (unsigned int)(i + 1);
      }
      goto LABEL_41;
    }
LABEL_57:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v6 - *(_DWORD *)(a1 + 20) - v24 <= v6 >> 3 )
  {
    sub_31E5EB0(a1, v6);
    v35 = *(_DWORD *)(a1 + 24);
    if ( v35 )
    {
      v36 = v35 - 1;
      v20 = 0;
      i = 1;
      for ( k = v36
              & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v5) | ((unsigned __int64)(unsigned int)(37 * v4) << 32))) >> 31)
               ^ (756364221 * v5)); ; k = v36 & v39 )
      {
        v37 = *(_QWORD *)(a1 + 8);
        v8 = (_DWORD *)(v37 + 12LL * k);
        if ( v4 == *v8 && v5 == v8[1] )
          break;
        if ( !*v8 )
        {
          v40 = v8[1];
          if ( v40 == -1 )
            goto LABEL_54;
          if ( v40 == -2 && !v20 )
            v20 = (_DWORD *)(v37 + 12LL * k);
        }
        v39 = i + k;
        i = (unsigned int)(i + 1);
      }
LABEL_41:
      v24 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_24;
    }
    goto LABEL_57;
  }
LABEL_24:
  *(_DWORD *)(a1 + 16) = v24;
  if ( *v8 || v8[1] != -1 )
    --*(_DWORD *)(a1 + 20);
  *v8 = v4;
  v8[1] = v5;
  v8[2] = 0;
  v25 = *(_QWORD *)a2;
  v26 = *(unsigned int *)(a1 + 44);
  v42[1] = 0;
  v42[0] = v25;
  v27 = *(unsigned int *)(a1 + 40);
  v42[2] = 0;
  v28 = v27 + 1;
  if ( v27 + 1 > v26 )
  {
    v32 = *(_QWORD *)(a1 + 32);
    v33 = a1 + 32;
    v34 = (const void *)(a1 + 48);
    if ( v32 > (unsigned __int64)v42 || (unsigned __int64)v42 >= v32 + 24 * v27 )
    {
      sub_C8D5F0(v33, v34, v28, 0x18u, i, v10);
      v29 = *(_QWORD *)(a1 + 32);
      v27 = *(unsigned int *)(a1 + 40);
      v30 = (const __m128i *)v42;
    }
    else
    {
      sub_C8D5F0(v33, v34, v28, 0x18u, i, v10);
      v29 = *(_QWORD *)(a1 + 32);
      v27 = *(unsigned int *)(a1 + 40);
      v30 = (const __m128i *)((char *)v42 + v29 - v32);
    }
  }
  else
  {
    v29 = *(_QWORD *)(a1 + 32);
    v30 = (const __m128i *)v42;
  }
  v31 = (__m128i *)(v29 + 24 * v27);
  *v31 = _mm_loadu_si128(v30);
  v31[1].m128i_i64[0] = v30[1].m128i_i64[0];
  v14 = *(unsigned int *)(a1 + 40);
  *(_DWORD *)(a1 + 40) = v14 + 1;
  v8[2] = v14;
  return *(_QWORD *)(a1 + 32) + 24 * v14 + 8;
}

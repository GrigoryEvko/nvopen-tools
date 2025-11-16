// Function: sub_1E4A230
// Address: 0x1e4a230
//
_DWORD *__fastcall sub_1E4A230(__int64 a1, int a2)
{
  __int64 v3; // rbx
  int *v4; // r15
  unsigned __int64 v5; // rax
  _DWORD *result; // rax
  int *v7; // r14
  _DWORD *i; // rdx
  int *v9; // r12
  int v10; // eax
  int v11; // edx
  int v12; // edx
  __int64 v13; // rdi
  int v14; // r9d
  unsigned int v15; // ecx
  int *v16; // r8
  int *v17; // rbx
  int v18; // esi
  __int64 v19; // r8
  __int64 v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // r15
  __int64 v26; // r11
  __int64 v27; // r10
  __int64 v28; // r9
  __int64 *v29; // rdi
  __int64 v30; // rdx
  _DWORD *j; // rdx
  int *v32; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(int **)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_DWORD *)sub_22077B0(88LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[22 * v3];
    for ( i = &result[22 * *(unsigned int *)(a1 + 24)]; i != result; result += 22 )
    {
      if ( result )
        *result = 0x7FFFFFFF;
    }
    if ( v7 != v4 )
    {
      v32 = v4;
      v9 = v4;
      do
      {
        while ( 1 )
        {
          v10 = *v9;
          if ( (unsigned int)(*v9 + 0x7FFFFFFF) <= 0xFFFFFFFD )
            break;
          v9 += 22;
          if ( v7 == v9 )
            goto LABEL_17;
        }
        v11 = *(_DWORD *)(a1 + 24);
        if ( !v11 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v12 = v11 - 1;
        v13 = *(_QWORD *)(a1 + 8);
        v14 = 1;
        v15 = v12 & (37 * v10);
        v16 = 0;
        v17 = (int *)(v13 + 88LL * v15);
        v18 = *v17;
        if ( v10 != *v17 )
        {
          while ( v18 != 0x7FFFFFFF )
          {
            if ( !v16 && v18 == 0x80000000 )
              v16 = v17;
            v15 = v12 & (v14 + v15);
            v17 = (int *)(v13 + 88LL * v15);
            v18 = *v17;
            if ( v10 == *v17 )
              goto LABEL_14;
            ++v14;
          }
          if ( v16 )
            v17 = v16;
        }
LABEL_14:
        *v17 = v10;
        *((_QWORD *)v17 + 1) = 0;
        *((_QWORD *)v17 + 2) = 0;
        *((_QWORD *)v17 + 3) = 0;
        *((_QWORD *)v17 + 4) = 0;
        *((_QWORD *)v17 + 5) = 0;
        *((_QWORD *)v17 + 6) = 0;
        *((_QWORD *)v17 + 7) = 0;
        *((_QWORD *)v17 + 8) = 0;
        *((_QWORD *)v17 + 9) = 0;
        *((_QWORD *)v17 + 10) = 0;
        sub_1E47CF0((__int64 *)v17 + 1, 0);
        if ( *((_QWORD *)v9 + 1) )
        {
          v19 = *((_QWORD *)v17 + 4);
          v20 = *((_QWORD *)v17 + 5);
          *((_QWORD *)v17 + 4) = 0;
          v21 = *((_QWORD *)v17 + 6);
          v22 = *((_QWORD *)v17 + 7);
          *((_QWORD *)v17 + 5) = 0;
          v23 = *((_QWORD *)v17 + 8);
          v24 = *((_QWORD *)v17 + 9);
          *((_QWORD *)v17 + 6) = 0;
          v25 = *((_QWORD *)v17 + 10);
          v26 = *((_QWORD *)v17 + 1);
          *((_QWORD *)v17 + 7) = 0;
          *((_QWORD *)v17 + 1) = 0;
          v27 = *((_QWORD *)v17 + 2);
          *((_QWORD *)v17 + 8) = 0;
          v28 = *((_QWORD *)v17 + 3);
          *((_QWORD *)v17 + 2) = 0;
          *((_QWORD *)v17 + 3) = 0;
          *((_QWORD *)v17 + 9) = 0;
          *((_QWORD *)v17 + 10) = 0;
          *(__m128i *)(v17 + 2) = _mm_loadu_si128((const __m128i *)(v9 + 2));
          *(__m128i *)(v17 + 6) = _mm_loadu_si128((const __m128i *)(v9 + 6));
          *(__m128i *)(v17 + 10) = _mm_loadu_si128((const __m128i *)(v9 + 10));
          *(__m128i *)(v17 + 14) = _mm_loadu_si128((const __m128i *)(v9 + 14));
          *(__m128i *)(v17 + 18) = _mm_loadu_si128((const __m128i *)(v9 + 18));
          *((_QWORD *)v9 + 1) = v26;
          *((_QWORD *)v9 + 2) = v27;
          *((_QWORD *)v9 + 3) = v28;
          *((_QWORD *)v9 + 4) = v19;
          *((_QWORD *)v9 + 5) = v20;
          *((_QWORD *)v9 + 6) = v21;
          *((_QWORD *)v9 + 7) = v22;
          *((_QWORD *)v9 + 8) = v23;
          *((_QWORD *)v9 + 9) = v24;
          *((_QWORD *)v9 + 10) = v25;
        }
        ++*(_DWORD *)(a1 + 16);
        v29 = (__int64 *)(v9 + 2);
        v9 += 22;
        sub_1E472E0(v29);
      }
      while ( v7 != v9 );
LABEL_17:
      v4 = v32;
    }
    return (_DWORD *)j___libc_free_0(v4);
  }
  else
  {
    v30 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[22 * v30]; j != result; result += 22 )
    {
      if ( result )
        *result = 0x7FFFFFFF;
    }
  }
  return result;
}

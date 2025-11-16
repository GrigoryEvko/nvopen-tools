// Function: sub_354BAE0
// Address: 0x354bae0
//
_DWORD *__fastcall sub_354BAE0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r15
  unsigned int v6; // eax
  _DWORD *result; // rax
  __int64 v8; // r14
  _DWORD *i; // rdx
  __int64 v10; // r12
  int v11; // eax
  int v12; // edx
  int v13; // edx
  __int64 v14; // rdi
  int v15; // r9d
  unsigned int v16; // ecx
  int *v17; // r8
  int *v18; // rbx
  int v19; // esi
  __int64 v20; // r8
  __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // r11
  __int64 v28; // r10
  __int64 v29; // r9
  unsigned __int64 *v30; // rdi
  __int64 v31; // rdx
  _DWORD *j; // rdx
  __int64 v33; // [rsp+0h] [rbp-40h]
  __int64 v34; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_DWORD *)sub_C7D670(88LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v34 = 88 * v4;
    v8 = v5 + 88 * v4;
    for ( i = &result[22 * *(unsigned int *)(a1 + 24)]; i != result; result += 22 )
    {
      if ( result )
        *result = 0x7FFFFFFF;
    }
    if ( v8 != v5 )
    {
      v33 = v5;
      v10 = v5;
      do
      {
        while ( 1 )
        {
          v11 = *(_DWORD *)v10;
          if ( (unsigned int)(*(_DWORD *)v10 + 0x7FFFFFFF) <= 0xFFFFFFFD )
            break;
          v10 += 88;
          if ( v8 == v10 )
            goto LABEL_17;
        }
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v13 = v12 - 1;
        v14 = *(_QWORD *)(a1 + 8);
        v15 = 1;
        v16 = v13 & (37 * v11);
        v17 = 0;
        v18 = (int *)(v14 + 88LL * v16);
        v19 = *v18;
        if ( v11 != *v18 )
        {
          while ( v19 != 0x7FFFFFFF )
          {
            if ( !v17 && v19 == 0x80000000 )
              v17 = v18;
            v16 = v13 & (v15 + v16);
            v18 = (int *)(v14 + 88LL * v16);
            v19 = *v18;
            if ( v11 == *v18 )
              goto LABEL_14;
            ++v15;
          }
          if ( v17 )
            v18 = v17;
        }
LABEL_14:
        *v18 = v11;
        *((_QWORD *)v18 + 1) = 0;
        *((_QWORD *)v18 + 2) = 0;
        *((_QWORD *)v18 + 3) = 0;
        *((_QWORD *)v18 + 4) = 0;
        *((_QWORD *)v18 + 5) = 0;
        *((_QWORD *)v18 + 6) = 0;
        *((_QWORD *)v18 + 7) = 0;
        *((_QWORD *)v18 + 8) = 0;
        *((_QWORD *)v18 + 9) = 0;
        *((_QWORD *)v18 + 10) = 0;
        sub_3547BF0((__int64 *)v18 + 1, 0);
        if ( *(_QWORD *)(v10 + 8) )
        {
          v20 = *((_QWORD *)v18 + 4);
          v21 = *((_QWORD *)v18 + 5);
          *((_QWORD *)v18 + 4) = 0;
          v22 = *((_QWORD *)v18 + 6);
          v23 = *((_QWORD *)v18 + 7);
          *((_QWORD *)v18 + 5) = 0;
          v24 = *((_QWORD *)v18 + 8);
          v25 = *((_QWORD *)v18 + 9);
          *((_QWORD *)v18 + 6) = 0;
          v26 = *((_QWORD *)v18 + 10);
          v27 = *((_QWORD *)v18 + 1);
          *((_QWORD *)v18 + 7) = 0;
          *((_QWORD *)v18 + 1) = 0;
          v28 = *((_QWORD *)v18 + 2);
          *((_QWORD *)v18 + 8) = 0;
          v29 = *((_QWORD *)v18 + 3);
          *((_QWORD *)v18 + 2) = 0;
          *((_QWORD *)v18 + 3) = 0;
          *((_QWORD *)v18 + 9) = 0;
          *((_QWORD *)v18 + 10) = 0;
          *(__m128i *)(v18 + 2) = _mm_loadu_si128((const __m128i *)(v10 + 8));
          *(__m128i *)(v18 + 6) = _mm_loadu_si128((const __m128i *)(v10 + 24));
          *(__m128i *)(v18 + 10) = _mm_loadu_si128((const __m128i *)(v10 + 40));
          *(__m128i *)(v18 + 14) = _mm_loadu_si128((const __m128i *)(v10 + 56));
          *(__m128i *)(v18 + 18) = _mm_loadu_si128((const __m128i *)(v10 + 72));
          *(_QWORD *)(v10 + 8) = v27;
          *(_QWORD *)(v10 + 16) = v28;
          *(_QWORD *)(v10 + 24) = v29;
          *(_QWORD *)(v10 + 32) = v20;
          *(_QWORD *)(v10 + 40) = v21;
          *(_QWORD *)(v10 + 48) = v22;
          *(_QWORD *)(v10 + 56) = v23;
          *(_QWORD *)(v10 + 64) = v24;
          *(_QWORD *)(v10 + 72) = v25;
          *(_QWORD *)(v10 + 80) = v26;
        }
        ++*(_DWORD *)(a1 + 16);
        v30 = (unsigned __int64 *)(v10 + 8);
        v10 += 88;
        sub_3546C50(v30);
      }
      while ( v8 != v10 );
LABEL_17:
      v5 = v33;
    }
    return (_DWORD *)sub_C7D6A0(v5, v34, 8);
  }
  else
  {
    v31 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[22 * v31]; j != result; result += 22 )
    {
      if ( result )
        *result = 0x7FFFFFFF;
    }
  }
  return result;
}

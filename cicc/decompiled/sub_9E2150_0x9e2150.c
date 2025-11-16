// Function: sub_9E2150
// Address: 0x9e2150
//
_QWORD *__fastcall sub_9E2150(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r13
  __int64 *v5; // r12
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 *v9; // rdi
  _QWORD *i; // rdx
  __int64 *v11; // rax
  unsigned __int64 v12; // rdx
  int v13; // esi
  int v14; // esi
  __int64 v15; // r11
  int v16; // r15d
  unsigned __int64 *v17; // r14
  unsigned int v18; // ecx
  unsigned __int64 *v19; // r8
  unsigned __int64 v20; // r10
  __m128i v21; // xmm0
  __int64 v22; // rdx
  _QWORD *j; // rdx

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(__int64 **)(a1 + 8);
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
  result = (_QWORD *)sub_C7D670(24LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = &v5[3 * v4];
    for ( i = &result[3 * v8]; i != result; result += 3 )
    {
      if ( result )
        *result = -1;
    }
    if ( v9 != v5 )
    {
      v11 = v5;
      do
      {
        while ( 1 )
        {
          v12 = *v11;
          if ( (unsigned __int64)*v11 <= 0xFFFFFFFFFFFFFFFDLL )
            break;
          v11 += 3;
          if ( v9 == v11 )
            return (_QWORD *)sub_C7D6A0(v5, 24 * v4, 8);
        }
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 1;
        v17 = 0;
        v18 = v14 & (((0xBF58476D1CE4E5B9LL * v12) >> 31) ^ (484763065 * v12));
        v19 = (unsigned __int64 *)(v15 + 24LL * v18);
        v20 = *v19;
        if ( v12 != *v19 )
        {
          while ( v20 != -1 )
          {
            if ( !v17 && v20 == -2 )
              v17 = v19;
            v18 = v14 & (v16 + v18);
            v19 = (unsigned __int64 *)(v15 + 24LL * v18);
            v20 = *v19;
            if ( v12 == *v19 )
              goto LABEL_14;
            ++v16;
          }
          if ( v17 )
            v19 = v17;
        }
LABEL_14:
        *v19 = v12;
        v21 = _mm_loadu_si128((const __m128i *)(v11 + 1));
        v11 += 3;
        *(__m128i *)(v19 + 1) = v21;
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v9 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v5, 24 * v4, 8);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[3 * v22]; j != result; result += 3 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}

// Function: sub_37BF080
// Address: 0x37bf080
//
_DWORD *__fastcall sub_37BF080(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  unsigned int v5; // eax
  _DWORD *result; // rax
  __int64 v7; // r8
  __int64 v8; // r13
  __int64 v9; // r14
  _DWORD *i; // rdx
  __int64 v11; // rbx
  int v12; // eax
  int v13; // ecx
  __int64 v14; // rcx
  __int64 v15; // r9
  int v16; // r11d
  __int64 v17; // r10
  unsigned int v18; // esi
  __int64 v19; // rdx
  int v20; // edi
  unsigned __int64 v21; // rdi
  __int64 v22; // rdx
  _DWORD *j; // rdx
  __int64 v24; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_DWORD *)sub_C7D670(88LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 88 * v3;
    v9 = v4 + 88 * v3;
    for ( i = &result[22 * *(unsigned int *)(a1 + 24)]; i != result; result += 22 )
    {
      if ( result )
        *result = -1;
    }
    if ( v9 != v4 )
    {
      v11 = v4;
      do
      {
        while ( 1 )
        {
          v12 = *(_DWORD *)v11;
          if ( *(_DWORD *)v11 <= 0xFFFFFFFD )
          {
            v13 = *(_DWORD *)(a1 + 24);
            if ( !v13 )
            {
              MEMORY[0] = 0;
              BUG();
            }
            v14 = (unsigned int)(v13 - 1);
            v15 = *(_QWORD *)(a1 + 8);
            v16 = 1;
            v17 = 0;
            v18 = v14 & (37 * v12);
            v19 = v15 + 88LL * v18;
            v20 = *(_DWORD *)v19;
            if ( *(_DWORD *)v19 != v12 )
            {
              while ( v20 != -1 )
              {
                if ( !v17 && v20 == -2 )
                  v17 = v19;
                v7 = (unsigned int)(v16 + 1);
                v18 = v14 & (v16 + v18);
                v19 = v15 + 88LL * v18;
                v20 = *(_DWORD *)v19;
                if ( v12 == *(_DWORD *)v19 )
                  goto LABEL_14;
                ++v16;
              }
              if ( v17 )
                v19 = v17;
            }
LABEL_14:
            *(_DWORD *)v19 = v12;
            *(_QWORD *)(v19 + 8) = v19 + 24;
            *(_QWORD *)(v19 + 16) = 0x100000000LL;
            if ( *(_DWORD *)(v11 + 16) )
            {
              v24 = v19;
              sub_37B6900(v19 + 8, (char **)(v11 + 8), v19, v14, v7, v15);
              v19 = v24;
            }
            *(__m128i *)(v19 + 72) = _mm_loadu_si128((const __m128i *)(v11 + 72));
            ++*(_DWORD *)(a1 + 16);
            v21 = *(_QWORD *)(v11 + 8);
            if ( v21 != v11 + 24 )
              break;
          }
          v11 += 88;
          if ( v9 == v11 )
            return (_DWORD *)sub_C7D6A0(v4, v8, 8);
        }
        _libc_free(v21);
        v11 += 88;
      }
      while ( v9 != v11 );
    }
    return (_DWORD *)sub_C7D6A0(v4, v8, 8);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[22 * v22]; j != result; result += 22 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}

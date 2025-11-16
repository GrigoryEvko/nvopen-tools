// Function: sub_248BD20
// Address: 0x248bd20
//
_QWORD *__fastcall sub_248BD20(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r13
  __int64 v5; // r14
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  _QWORD *i; // rdx
  __int64 v11; // r15
  __int64 v12; // rax
  int v13; // esi
  int v14; // esi
  __int64 v15; // r9
  int v16; // r11d
  __int64 *v17; // r10
  unsigned int v18; // edx
  __int64 *v19; // rdi
  __int64 v20; // r8
  _QWORD *v21; // r13
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  __int64 v24; // rdx
  _QWORD *j; // rdx
  __int64 v26; // [rsp+8h] [rbp-38h]

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
  result = (_QWORD *)sub_C7D670((unsigned __int64)v6 << 6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v26 = v4 << 6;
    v9 = v5 + (v4 << 6);
    for ( i = &result[8 * v8]; i != result; result += 8 )
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
          v12 = *(_QWORD *)v11;
          if ( *(_QWORD *)v11 <= 0xFFFFFFFFFFFFFFFDLL )
          {
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
            v19 = (__int64 *)(v15 + ((unsigned __int64)v18 << 6));
            v20 = *v19;
            if ( *v19 != v12 )
            {
              while ( v20 != -1 )
              {
                if ( !v17 && v20 == -2 )
                  v17 = v19;
                v18 = v14 & (v16 + v18);
                v19 = (__int64 *)(v15 + ((unsigned __int64)v18 << 6));
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
            sub_248B5F0((__m128i *)(v19 + 1), (__m128i *)(v11 + 8));
            ++*(_DWORD *)(a1 + 16);
            v21 = *(_QWORD **)(v11 + 24);
            while ( v21 )
            {
              v22 = (unsigned __int64)v21;
              v21 = (_QWORD *)*v21;
              j_j___libc_free_0(v22);
            }
            memset(*(void **)(v11 + 8), 0, 8LL * *(_QWORD *)(v11 + 16));
            v23 = *(_QWORD *)(v11 + 8);
            *(_QWORD *)(v11 + 32) = 0;
            *(_QWORD *)(v11 + 24) = 0;
            if ( v23 != v11 + 56 )
              break;
          }
          v11 += 64;
          if ( v9 == v11 )
            return (_QWORD *)sub_C7D6A0(v5, v26, 8);
        }
        v11 += 64;
        j_j___libc_free_0(v23);
      }
      while ( v9 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v26, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[8 * v24]; j != result; result += 8 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}

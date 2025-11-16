// Function: sub_2604730
// Address: 0x2604730
//
_DWORD *__fastcall sub_2604730(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _DWORD *result; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // r14
  _DWORD *i; // rdx
  __int64 v13; // rbx
  unsigned int v14; // eax
  int v15; // edx
  __int64 v16; // rdx
  __int64 v17; // r9
  int v18; // r11d
  unsigned int *v19; // r10
  unsigned int v20; // esi
  unsigned int *v21; // rdi
  __int64 v22; // r8
  unsigned __int64 v23; // rdi
  __int64 v24; // rdx
  _DWORD *j; // rdx

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
  result = (_DWORD *)sub_C7D670(40LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v9 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v10 = 40 * v4;
    v11 = v5 + 40 * v4;
    for ( i = &result[10 * v9]; i != result; result += 10 )
    {
      if ( result )
        *result = -1;
    }
    if ( v11 != v5 )
    {
      v13 = v5;
      do
      {
        while ( 1 )
        {
          v14 = *(_DWORD *)v13;
          if ( *(_DWORD *)v13 <= 0xFFFFFFFD )
          {
            v15 = *(_DWORD *)(a1 + 24);
            if ( !v15 )
            {
              MEMORY[0] = 0;
              BUG();
            }
            v16 = (unsigned int)(v15 - 1);
            v17 = *(_QWORD *)(a1 + 8);
            v18 = 1;
            v19 = 0;
            v20 = v16 & (37 * v14);
            v21 = (unsigned int *)(v17 + 40LL * v20);
            v22 = *v21;
            if ( v14 != (_DWORD)v22 )
            {
              while ( (_DWORD)v22 != -1 )
              {
                if ( !v19 && (_DWORD)v22 == -2 )
                  v19 = v21;
                v8 = (unsigned int)(v18 + 1);
                v20 = v16 & (v18 + v20);
                v21 = (unsigned int *)(v17 + 40LL * v20);
                v22 = *v21;
                if ( v14 == (_DWORD)v22 )
                  goto LABEL_14;
                ++v18;
              }
              if ( v19 )
                v21 = v19;
            }
LABEL_14:
            *v21 = v14;
            *((_QWORD *)v21 + 1) = *(_QWORD *)(v13 + 8);
            *((_QWORD *)v21 + 2) = v21 + 8;
            *((_QWORD *)v21 + 3) = 0x200000000LL;
            if ( *(_DWORD *)(v13 + 24) )
              sub_25F5FB0((__int64)(v21 + 4), (char **)(v13 + 16), v16, v8, v22, v17);
            ++*(_DWORD *)(a1 + 16);
            v23 = *(_QWORD *)(v13 + 16);
            if ( v23 != v13 + 32 )
              break;
          }
          v13 += 40;
          if ( v11 == v13 )
            return (_DWORD *)sub_C7D6A0(v5, v10, 8);
        }
        _libc_free(v23);
        v13 += 40;
      }
      while ( v11 != v13 );
    }
    return (_DWORD *)sub_C7D6A0(v5, v10, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[10 * v24]; j != result; result += 10 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}

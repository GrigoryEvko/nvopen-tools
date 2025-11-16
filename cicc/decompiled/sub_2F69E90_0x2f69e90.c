// Function: sub_2F69E90
// Address: 0x2f69e90
//
_DWORD *__fastcall sub_2F69E90(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r12
  __int64 v5; // r13
  unsigned int v6; // edi
  _DWORD *result; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 v11; // r14
  _DWORD *i; // rdx
  char **v13; // rbx
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
  result = (_DWORD *)sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v9 = *(unsigned int *)(a1 + 24);
    v10 = 32 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v11 = v5 + v10;
    for ( i = &result[8 * v9]; i != result; result += 8 )
    {
      if ( result )
        *result = -1;
    }
    if ( v11 != v5 )
    {
      v13 = (char **)v5;
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
              MEMORY[0] = *(_DWORD *)v13;
              BUG();
            }
            v16 = (unsigned int)(v15 - 1);
            v17 = *(_QWORD *)(a1 + 8);
            v18 = 1;
            v19 = 0;
            v20 = v16 & (37 * v14);
            v21 = (unsigned int *)(v17 + 32LL * v20);
            v22 = *v21;
            if ( v14 != (_DWORD)v22 )
            {
              while ( (_DWORD)v22 != -1 )
              {
                if ( !v19 && (_DWORD)v22 == -2 )
                  v19 = v21;
                v8 = (unsigned int)(v18 + 1);
                v20 = v16 & (v18 + v20);
                v21 = (unsigned int *)(v17 + 32LL * v20);
                v22 = *v21;
                if ( v14 == (_DWORD)v22 )
                  goto LABEL_14;
                ++v18;
              }
              if ( v19 )
                v21 = v19;
            }
LABEL_14:
            *v21 = *(_DWORD *)v13;
            *((_QWORD *)v21 + 1) = v21 + 6;
            *((_QWORD *)v21 + 2) = 0x200000000LL;
            if ( *((_DWORD *)v13 + 4) )
              sub_2F61140((__int64)(v21 + 2), v13 + 1, v16, v8, v22, v17);
            ++*(_DWORD *)(a1 + 16);
            v23 = (unsigned __int64)v13[1];
            if ( (char **)v23 != v13 + 3 )
              break;
          }
          v13 += 4;
          if ( (char **)v11 == v13 )
            return (_DWORD *)sub_C7D6A0(v5, v10, 8);
        }
        _libc_free(v23);
        v13 += 4;
      }
      while ( (char **)v11 != v13 );
    }
    return (_DWORD *)sub_C7D6A0(v5, v10, 8);
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

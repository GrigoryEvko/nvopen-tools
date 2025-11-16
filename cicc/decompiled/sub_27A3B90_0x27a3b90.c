// Function: sub_27A3B90
// Address: 0x27a3b90
//
__int64 __fastcall sub_27A3B90(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // edi
  __int64 result; // rax
  __int64 v8; // rdx
  unsigned int *v9; // r12
  __int64 i; // rdx
  unsigned int *j; // rbx
  __int64 v12; // rdx
  int v13; // edi
  __int64 v14; // rcx
  __int64 v15; // rsi
  int v16; // r10d
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned int k; // eax
  __int64 v20; // rdi
  int v21; // r11d
  unsigned __int64 v22; // rdi
  unsigned int v23; // eax
  __int64 v24; // rdx
  __int64 m; // rdx
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
  result = sub_C7D670((unsigned __int64)v6 << 6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v26 = v4 << 6;
    v9 = (unsigned int *)(v5 + (v4 << 6));
    for ( i = result + (v8 << 6); i != result; result += 64 )
    {
      if ( result )
      {
        *(_DWORD *)result = -1;
        *(_QWORD *)(result + 8) = -1;
      }
    }
    if ( v9 != (unsigned int *)v5 )
    {
      for ( j = (unsigned int *)v5; v9 != j; j += 16 )
      {
        while ( 1 )
        {
          v12 = *j;
          if ( (_DWORD)v12 != -1 )
            break;
          if ( *((_QWORD *)j + 1) == -1 )
          {
LABEL_22:
            j += 16;
            if ( v9 == j )
              return sub_C7D6A0(v5, v26, 8);
          }
          else
          {
LABEL_12:
            v13 = *(_DWORD *)(a1 + 24);
            if ( !v13 )
            {
              MEMORY[0] = 0;
              BUG();
            }
            v14 = *((_QWORD *)j + 1);
            v15 = *(_QWORD *)(a1 + 8);
            v16 = 1;
            v17 = (unsigned int)(v13 - 1);
            v18 = 0;
            for ( k = v17
                    & (((0xBF58476D1CE4E5B9LL
                       * ((unsigned int)((0xBF58476D1CE4E5B9LL * v14) >> 31) ^ (484763065 * j[2])
                        | ((unsigned __int64)(unsigned int)(37 * v12) << 32))) >> 31)
                     ^ (484763065 * (((0xBF58476D1CE4E5B9LL * v14) >> 31) ^ (484763065 * j[2])))); ; k = v17 & v23 )
            {
              v20 = v15 + ((unsigned __int64)k << 6);
              v21 = *(_DWORD *)v20;
              if ( (_DWORD)v12 == *(_DWORD *)v20 && *(_QWORD *)(v20 + 8) == v14 )
                break;
              if ( v21 == -1 )
              {
                if ( *(_QWORD *)(v20 + 8) == -1 )
                {
                  if ( v18 )
                    v20 = v18;
                  break;
                }
              }
              else if ( v21 == -2 && *(_QWORD *)(v20 + 8) == -2 && !v18 )
              {
                v18 = v15 + ((unsigned __int64)k << 6);
              }
              v23 = v16 + k;
              ++v16;
            }
            *(_DWORD *)v20 = v12;
            *(_QWORD *)(v20 + 8) = *((_QWORD *)j + 1);
            *(_QWORD *)(v20 + 16) = v20 + 32;
            *(_QWORD *)(v20 + 24) = 0x400000000LL;
            if ( j[6] )
              sub_27A0EB0(v20 + 16, (char **)j + 2, v12, v14, v17, v18);
            ++*(_DWORD *)(a1 + 16);
            v22 = *((_QWORD *)j + 2);
            if ( (unsigned int *)v22 == j + 8 )
              goto LABEL_22;
            _libc_free(v22);
            j += 16;
            if ( v9 == j )
              return sub_C7D6A0(v5, v26, 8);
          }
        }
        if ( (_DWORD)v12 != -2 || *((_QWORD *)j + 1) != -2 )
          goto LABEL_12;
      }
    }
    return sub_C7D6A0(v5, v26, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = result + (v24 << 6); m != result; result += 64 )
    {
      if ( result )
      {
        *(_DWORD *)result = -1;
        *(_QWORD *)(result + 8) = -1;
      }
    }
  }
  return result;
}

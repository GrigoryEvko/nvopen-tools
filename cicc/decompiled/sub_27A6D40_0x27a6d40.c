// Function: sub_27A6D40
// Address: 0x27a6d40
//
__int64 __fastcall sub_27A6D40(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // eax
  __int64 result; // rax
  unsigned int *v8; // r12
  __int64 i; // rdx
  unsigned int *j; // rbx
  __int64 v11; // rdx
  int v12; // edi
  __int64 v13; // rcx
  __int64 v14; // rsi
  int v15; // r10d
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned int k; // eax
  __int64 v19; // rdi
  int v20; // r11d
  unsigned __int64 v21; // rdi
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int64 m; // rdx
  __int64 v25; // [rsp+8h] [rbp-38h]

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
  result = sub_C7D670(48LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v25 = 48 * v4;
    v8 = (unsigned int *)(v5 + 48 * v4);
    for ( i = result + 48LL * *(unsigned int *)(a1 + 24); i != result; result += 48 )
    {
      if ( result )
      {
        *(_DWORD *)result = -1;
        *(_QWORD *)(result + 8) = -1;
      }
    }
    if ( v8 != (unsigned int *)v5 )
    {
      for ( j = (unsigned int *)v5; v8 != j; j += 12 )
      {
        while ( 1 )
        {
          v11 = *j;
          if ( (_DWORD)v11 != -1 )
            break;
          if ( *((_QWORD *)j + 1) == -1 )
          {
LABEL_22:
            j += 12;
            if ( v8 == j )
              return sub_C7D6A0(v5, v25, 8);
          }
          else
          {
LABEL_12:
            v12 = *(_DWORD *)(a1 + 24);
            if ( !v12 )
            {
              MEMORY[0] = 0;
              BUG();
            }
            v13 = *((_QWORD *)j + 1);
            v14 = *(_QWORD *)(a1 + 8);
            v15 = 1;
            v16 = (unsigned int)(v12 - 1);
            v17 = 0;
            for ( k = v16
                    & (((0xBF58476D1CE4E5B9LL
                       * ((unsigned int)((0xBF58476D1CE4E5B9LL * v13) >> 31) ^ (484763065 * j[2])
                        | ((unsigned __int64)(unsigned int)(37 * v11) << 32))) >> 31)
                     ^ (484763065 * (((0xBF58476D1CE4E5B9LL * v13) >> 31) ^ (484763065 * j[2])))); ; k = v16 & v22 )
            {
              v19 = v14 + 48LL * k;
              v20 = *(_DWORD *)v19;
              if ( (_DWORD)v11 == *(_DWORD *)v19 && *(_QWORD *)(v19 + 8) == v13 )
                break;
              if ( v20 == -1 )
              {
                if ( *(_QWORD *)(v19 + 8) == -1 )
                {
                  if ( v17 )
                    v19 = v17;
                  break;
                }
              }
              else if ( v20 == -2 && *(_QWORD *)(v19 + 8) == -2 && !v17 )
              {
                v17 = v14 + 48LL * k;
              }
              v22 = v15 + k;
              ++v15;
            }
            *(_DWORD *)v19 = v11;
            *(_QWORD *)(v19 + 8) = *((_QWORD *)j + 1);
            *(_QWORD *)(v19 + 16) = v19 + 32;
            *(_QWORD *)(v19 + 24) = 0x200000000LL;
            if ( j[6] )
              sub_27A0EB0(v19 + 16, (char **)j + 2, v11, v13, v16, v17);
            ++*(_DWORD *)(a1 + 16);
            v21 = *((_QWORD *)j + 2);
            if ( (unsigned int *)v21 == j + 8 )
              goto LABEL_22;
            _libc_free(v21);
            j += 12;
            if ( v8 == j )
              return sub_C7D6A0(v5, v25, 8);
          }
        }
        if ( (_DWORD)v11 != -2 || *((_QWORD *)j + 1) != -2 )
          goto LABEL_12;
      }
    }
    return sub_C7D6A0(v5, v25, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = result + 48 * v23; m != result; result += 48 )
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

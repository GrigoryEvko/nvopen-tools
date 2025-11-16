// Function: sub_2A6E970
// Address: 0x2a6e970
//
__int64 __fastcall sub_2A6E970(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r13
  __int64 v5; // r12
  unsigned int v6; // eax
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // r9
  __int64 i; // rdx
  __int64 v11; // rdx
  __int64 v12; // rcx
  int v13; // r8d
  int v14; // esi
  __int64 v15; // rdi
  __int64 *v16; // r14
  int v17; // r10d
  int v18; // r11d
  unsigned int j; // eax
  __int64 *v20; // r8
  __int64 v21; // r15
  int v22; // eax
  unsigned int v23; // eax
  __int64 v24; // rdx
  __int64 k; // rdx

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
  result = sub_C7D670(24LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = v5 + 24 * v4;
    for ( i = result + 24 * v8; i != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_DWORD *)(result + 8) = -1;
      }
    }
    if ( v9 != v5 )
    {
      v11 = v5;
      while ( 1 )
      {
        v12 = *(_QWORD *)v11;
        if ( *(_QWORD *)v11 == -4096 )
        {
          if ( *(_DWORD *)(v11 + 8) != -1 )
            goto LABEL_12;
          v11 += 24;
          if ( v9 == v11 )
            return sub_C7D6A0(v5, 24 * v4, 8);
        }
        else if ( v12 == -8192 && *(_DWORD *)(v11 + 8) == -2 )
        {
          v11 += 24;
          if ( v9 == v11 )
            return sub_C7D6A0(v5, 24 * v4, 8);
        }
        else
        {
LABEL_12:
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = *(_QWORD *)v11;
            BUG();
          }
          v14 = *(_DWORD *)(v11 + 8);
          v16 = 0;
          v17 = v13 - 1;
          v18 = 1;
          for ( j = (v13 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * ((unsigned int)(37 * v14)
                      | ((unsigned __int64)(((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)) << 32))) >> 31)
                   ^ (756364221 * v14)); ; j = v17 & v23 )
          {
            v15 = *(_QWORD *)(a1 + 8);
            v20 = (__int64 *)(v15 + 24LL * j);
            v21 = *v20;
            if ( v12 == *v20 && *((_DWORD *)v20 + 2) == v14 )
              break;
            if ( v21 == -4096 )
            {
              if ( *((_DWORD *)v20 + 2) == -1 )
              {
                if ( v16 )
                  v20 = v16;
                break;
              }
            }
            else if ( v21 == -8192 && *((_DWORD *)v20 + 2) == -2 && !v16 )
            {
              v16 = (__int64 *)(v15 + 24LL * j);
            }
            v23 = v18 + j;
            ++v18;
          }
          *v20 = v12;
          v22 = *(_DWORD *)(v11 + 8);
          v11 += 24;
          *((_DWORD *)v20 + 2) = v22;
          *((_DWORD *)v20 + 4) = *(_DWORD *)(v11 - 8);
          ++*(_DWORD *)(a1 + 16);
          if ( v9 == v11 )
            return sub_C7D6A0(v5, 24 * v4, 8);
        }
      }
    }
    return sub_C7D6A0(v5, 24 * v4, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 24 * v24; k != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_DWORD *)(result + 8) = -1;
      }
    }
  }
  return result;
}

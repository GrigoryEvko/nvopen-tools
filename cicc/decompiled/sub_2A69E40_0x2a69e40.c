// Function: sub_2A69E40
// Address: 0x2a69e40
//
__int64 __fastcall sub_2A69E40(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // eax
  __int64 result; // rax
  __int64 v8; // rcx
  __int64 v9; // r13
  __int64 i; // rdx
  __int64 j; // rbx
  __int64 v12; // rcx
  int v13; // r8d
  int v14; // edx
  int v15; // r8d
  __int64 v16; // rsi
  __int64 *v17; // r10
  int v18; // r9d
  unsigned int k; // eax
  __int64 *v20; // rdi
  __int64 v21; // r11
  int v22; // eax
  unsigned __int8 v23; // al
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  int v26; // eax
  unsigned int v27; // eax
  __int64 m; // rdx
  __int64 v29; // [rsp+8h] [rbp-38h]

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
  result = sub_C7D670(56LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v29 = 56 * v4;
    v9 = v5 + 56 * v4;
    for ( i = result + 56 * v8; i != result; result += 56 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_DWORD *)(result + 8) = -1;
      }
    }
    if ( v9 != v5 )
    {
      for ( j = v5; v9 != j; j += 56 )
      {
        while ( 1 )
        {
          v12 = *(_QWORD *)j;
          if ( *(_QWORD *)j != -4096 )
            break;
          if ( *(_DWORD *)(j + 8) == -1 )
          {
LABEL_22:
            j += 56;
            if ( v9 == j )
              return sub_C7D6A0(v5, v29, 8);
          }
          else
          {
LABEL_12:
            v13 = *(_DWORD *)(a1 + 24);
            if ( !v13 )
            {
              MEMORY[0] = *(_QWORD *)j;
              BUG();
            }
            v14 = *(_DWORD *)(j + 8);
            v15 = v13 - 1;
            v16 = *(_QWORD *)(a1 + 8);
            v17 = 0;
            v18 = 1;
            for ( k = v15
                    & (((0xBF58476D1CE4E5B9LL
                       * ((unsigned int)(37 * v14)
                        | ((unsigned __int64)(((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)) << 32))) >> 31)
                     ^ (756364221 * v14)); ; k = v15 & v27 )
            {
              v20 = (__int64 *)(v16 + 56LL * k);
              v21 = *v20;
              if ( v12 == *v20 && *((_DWORD *)v20 + 2) == v14 )
                break;
              if ( v21 == -4096 )
              {
                if ( *((_DWORD *)v20 + 2) == -1 )
                {
                  if ( v17 )
                    v20 = v17;
                  break;
                }
              }
              else if ( v21 == -8192 && *((_DWORD *)v20 + 2) == -2 && !v17 )
              {
                v17 = (__int64 *)(v16 + 56LL * k);
              }
              v27 = v18 + k;
              ++v18;
            }
            *v20 = v12;
            v22 = *(_DWORD *)(j + 8);
            *((_BYTE *)v20 + 17) = 0;
            *((_DWORD *)v20 + 2) = v22;
            v23 = *(_BYTE *)(j + 16);
            *((_BYTE *)v20 + 16) = v23;
            if ( v23 > 3u )
            {
              if ( (unsigned __int8)(v23 - 4) <= 1u )
              {
                *((_DWORD *)v20 + 8) = *(_DWORD *)(j + 32);
                v20[3] = *(_QWORD *)(j + 24);
                v26 = *(_DWORD *)(j + 48);
                *(_DWORD *)(j + 32) = 0;
                *((_DWORD *)v20 + 12) = v26;
                v20[5] = *(_QWORD *)(j + 40);
                LOBYTE(v26) = *(_BYTE *)(j + 17);
                *(_DWORD *)(j + 48) = 0;
                *((_BYTE *)v20 + 17) = v26;
              }
            }
            else if ( v23 > 1u )
            {
              v20[3] = *(_QWORD *)(j + 24);
            }
            *(_BYTE *)(j + 16) = 0;
            ++*(_DWORD *)(a1 + 16);
            if ( (unsigned int)*(unsigned __int8 *)(j + 16) - 4 > 1 )
              goto LABEL_22;
            if ( *(_DWORD *)(j + 48) > 0x40u )
            {
              v24 = *(_QWORD *)(j + 40);
              if ( v24 )
                j_j___libc_free_0_0(v24);
            }
            if ( *(_DWORD *)(j + 32) <= 0x40u )
              goto LABEL_22;
            v25 = *(_QWORD *)(j + 24);
            if ( !v25 )
              goto LABEL_22;
            j_j___libc_free_0_0(v25);
            j += 56;
            if ( v9 == j )
              return sub_C7D6A0(v5, v29, 8);
          }
        }
        if ( v12 != -8192 || *(_DWORD *)(j + 8) != -2 )
          goto LABEL_12;
      }
    }
    return sub_C7D6A0(v5, v29, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = result + 56LL * *(unsigned int *)(a1 + 24); m != result; result += 56 )
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

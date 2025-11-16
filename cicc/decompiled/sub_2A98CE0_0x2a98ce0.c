// Function: sub_2A98CE0
// Address: 0x2a98ce0
//
__int64 __fastcall sub_2A98CE0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // r12d
  __int64 v5; // r15
  unsigned int v6; // edi
  __int64 result; // rax
  __int64 v8; // rdx
  char *v9; // r10
  __int64 i; // rdx
  char *v11; // rdx
  int v12; // ecx
  int v13; // edi
  int v14; // r8d
  int v15; // edi
  int v16; // r14d
  __int64 v17; // r13
  __int64 v18; // r15
  unsigned int j; // eax
  __int64 v20; // rsi
  int v21; // r11d
  int v22; // eax
  unsigned int v23; // eax
  __int64 v24; // rdx
  __int64 k; // rdx
  __int64 v26; // [rsp+0h] [rbp-40h]
  __int64 v27; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_DWORD *)(a1 + 24);
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
  result = sub_C7D670(16LL * v6, 4);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v27 = 16LL * v4;
    v9 = (char *)(v5 + v27);
    for ( i = result + 16 * v8; i != result; result += 16 )
    {
      if ( result )
      {
        *(_BYTE *)result = -1;
        *(_DWORD *)(result + 4) = -1;
        *(_DWORD *)(result + 8) = -1;
      }
    }
    if ( v9 != (char *)v5 )
    {
      v26 = v5;
      v11 = (char *)v5;
      while ( 1 )
      {
        while ( 1 )
        {
          v12 = *((_DWORD *)v11 + 2);
          if ( v12 != -1 )
            break;
          if ( *((_DWORD *)v11 + 1) == -1 && *v11 == -1 )
          {
            v11 += 16;
            if ( v9 == v11 )
              goto LABEL_26;
          }
          else
          {
LABEL_13:
            v13 = *(_DWORD *)(a1 + 24);
            if ( !v13 )
            {
              MEMORY[8] = 0;
              BUG();
            }
            v14 = *((_DWORD *)v11 + 1);
            v15 = v13 - 1;
            v16 = 1;
            v17 = *(_QWORD *)(a1 + 8);
            v18 = 0;
            for ( j = v15
                    & (((0xBF58476D1CE4E5B9LL
                       * (((unsigned __int64)(unsigned int)(37 * v12) << 32)
                        | (unsigned int)((0xBF58476D1CE4E5B9LL
                                        * ((969526130LL * (unsigned int)(37 * *v11)) & 0xFFFFFFFELL
                                         | ((unsigned __int64)(unsigned int)(37 * v14) << 32))) >> 31)
                        ^ (-1747130070 * *v11))) >> 31)
                     ^ (484763065
                      * (((0xBF58476D1CE4E5B9LL
                         * ((969526130LL * (unsigned int)(37 * *v11)) & 0xFFFFFFFELL
                          | ((unsigned __int64)(unsigned int)(37 * v14) << 32))) >> 31)
                       ^ (-1747130070 * *v11)))); ; j = v15 & v23 )
            {
              v20 = v17 + 16LL * j;
              v21 = *(_DWORD *)(v20 + 8);
              if ( v12 == v21 && v14 == *(_DWORD *)(v20 + 4) && *v11 == *(_BYTE *)v20 )
                break;
              if ( v21 == -1 )
              {
                if ( *(_DWORD *)(v20 + 4) == -1 && *(_BYTE *)v20 == 0xFF )
                {
                  if ( v18 )
                    v20 = v18;
                  break;
                }
              }
              else if ( v21 == -2 && *(_DWORD *)(v20 + 4) == -2 && *(_BYTE *)v20 == 0xFE && !v18 )
              {
                v18 = v17 + 16LL * j;
              }
              v23 = v16 + j;
              ++v16;
            }
            *(_DWORD *)(v20 + 8) = v12;
            v22 = *((_DWORD *)v11 + 1);
            v11 += 16;
            *(_DWORD *)(v20 + 4) = v22;
            *(_BYTE *)v20 = *(v11 - 16);
            *(_DWORD *)(v20 + 12) = *((_DWORD *)v11 - 1);
            ++*(_DWORD *)(a1 + 16);
            if ( v9 == v11 )
            {
LABEL_26:
              v5 = v26;
              return sub_C7D6A0(v5, v27, 4);
            }
          }
        }
        if ( v12 != -2 || *((_DWORD *)v11 + 1) != -2 || *v11 != -2 )
          goto LABEL_13;
        v11 += 16;
        if ( v9 == v11 )
          goto LABEL_26;
      }
    }
    return sub_C7D6A0(v5, v27, 4);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 16 * v24; k != result; result += 16 )
    {
      if ( result )
      {
        *(_BYTE *)result = -1;
        *(_DWORD *)(result + 4) = -1;
        *(_DWORD *)(result + 8) = -1;
      }
    }
  }
  return result;
}

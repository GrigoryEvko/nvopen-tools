// Function: sub_2579470
// Address: 0x2579470
//
__int64 __fastcall sub_2579470(__int64 a1, int a2)
{
  __int64 v3; // r12
  __int64 v4; // r15
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // r9
  __int64 i; // rdx
  __int64 v10; // rdx
  __int64 v11; // rcx
  int v12; // esi
  __int64 v13; // r8
  unsigned __int8 v14; // r12
  int v15; // esi
  __int64 v16; // r13
  int v17; // r14d
  __int64 *v18; // r15
  unsigned int j; // eax
  __int64 *v20; // rdi
  __int64 v21; // r11
  __int64 v22; // rax
  unsigned int v23; // eax
  __int64 v24; // rdx
  __int64 k; // rdx
  __int64 v26; // [rsp+0h] [rbp-40h]
  __int64 v27; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = sub_C7D670(24LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v27 = 24 * v3;
    v8 = v4 + 24 * v3;
    for ( i = result + 24 * v7; i != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_QWORD *)(result + 8) = -4096;
        *(_BYTE *)(result + 16) = -1;
      }
    }
    if ( v8 != v4 )
    {
      v26 = v4;
      v10 = v4;
      while ( 1 )
      {
        while ( 1 )
        {
          v11 = *(_QWORD *)v10;
          if ( *(_QWORD *)v10 != -4096 )
            break;
          if ( *(_QWORD *)(v10 + 8) == -4096 && *(_BYTE *)(v10 + 16) == 0xFF )
          {
            v10 += 24;
            if ( v8 == v10 )
              goto LABEL_25;
          }
          else
          {
LABEL_12:
            v12 = *(_DWORD *)(a1 + 24);
            if ( !v12 )
            {
              MEMORY[0] = *(_QWORD *)v10;
              BUG();
            }
            v13 = *(_QWORD *)(v10 + 8);
            v14 = *(_BYTE *)(v10 + 16);
            v15 = v12 - 1;
            v16 = *(_QWORD *)(a1 + 8);
            v17 = 1;
            v18 = 0;
            for ( j = v15
                    & (((0xBF58476D1CE4E5B9LL
                       * ((37 * (unsigned int)v14)
                        | ((((0xBF58476D1CE4E5B9LL
                            * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)
                             | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))) >> 31)
                          ^ (0xBF58476D1CE4E5B9LL
                           * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)
                            | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32)))) << 32))) >> 31)
                     ^ (756364221 * v14)); ; j = v15 & v23 )
            {
              v20 = (__int64 *)(v16 + 24LL * j);
              v21 = *v20;
              if ( v11 == *v20 && v13 == v20[1] && v14 == *((_BYTE *)v20 + 16) )
                break;
              if ( v21 == -4096 )
              {
                if ( v20[1] == -4096 && *((_BYTE *)v20 + 16) == 0xFF )
                {
                  if ( v18 )
                    v20 = v18;
                  break;
                }
              }
              else if ( v21 == -8192 && v20[1] == -8192 && *((_BYTE *)v20 + 16) == 0xFE && !v18 )
              {
                v18 = (__int64 *)(v16 + 24LL * j);
              }
              v23 = v17 + j;
              ++v17;
            }
            *v20 = v11;
            v22 = *(_QWORD *)(v10 + 8);
            v10 += 24;
            v20[1] = v22;
            *((_BYTE *)v20 + 16) = *(_BYTE *)(v10 - 8);
            ++*(_DWORD *)(a1 + 16);
            if ( v8 == v10 )
            {
LABEL_25:
              v4 = v26;
              return sub_C7D6A0(v4, v27, 8);
            }
          }
        }
        if ( v11 != -8192 || *(_QWORD *)(v10 + 8) != -8192 || *(_BYTE *)(v10 + 16) != 0xFE )
          goto LABEL_12;
        v10 += 24;
        if ( v8 == v10 )
          goto LABEL_25;
      }
    }
    return sub_C7D6A0(v4, v27, 8);
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
        *(_QWORD *)(result + 8) = -4096;
        *(_BYTE *)(result + 16) = -1;
      }
    }
  }
  return result;
}

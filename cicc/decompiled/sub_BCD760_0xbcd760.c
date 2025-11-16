// Function: sub_BCD760
// Address: 0xbcd760
//
__int64 __fastcall sub_BCD760(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r13
  __int64 *v5; // r12
  unsigned int v6; // eax
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 *v9; // r9
  __int64 i; // rdx
  __int64 *v11; // rdx
  __int64 v12; // rcx
  int v13; // edi
  int v14; // r8d
  char v15; // r12
  __int64 v16; // rsi
  int v17; // edi
  __int64 v18; // r11
  int v19; // r13d
  __int64 *v20; // r14
  unsigned int j; // eax
  __int64 *v22; // rsi
  __int64 v23; // r10
  unsigned int v24; // eax
  __int64 v25; // rdx
  __int64 k; // rdx
  __int64 *v27; // [rsp+0h] [rbp-40h]
  __int64 v28; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(__int64 **)(a1 + 8);
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
    v28 = 24 * v4;
    v9 = &v5[3 * v4];
    for ( i = result + 24 * v8; i != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_DWORD *)(result + 8) = -1;
        *(_BYTE *)(result + 12) = 1;
      }
    }
    if ( v9 != v5 )
    {
      v27 = v5;
      v11 = v5;
      while ( 1 )
      {
        v12 = *v11;
        if ( *v11 == -4096 )
        {
          if ( *((_DWORD *)v11 + 2) != -1 )
            goto LABEL_12;
          if ( !*((_BYTE *)v11 + 12) )
          {
            v13 = *(_DWORD *)(a1 + 24);
            if ( !v13 )
              goto LABEL_49;
            v14 = -1;
            v16 = 4294967259LL;
            v15 = 0;
            goto LABEL_14;
          }
        }
        else
        {
          if ( v12 != -8192 || *((_DWORD *)v11 + 2) != -2 )
          {
LABEL_12:
            v13 = *(_DWORD *)(a1 + 24);
            if ( !v13 )
              goto LABEL_49;
            v14 = *((_DWORD *)v11 + 2);
            v15 = *((_BYTE *)v11 + 12);
            v16 = (unsigned int)(37 * v14);
            if ( !v15 )
              goto LABEL_14;
            goto LABEL_37;
          }
          if ( *((_BYTE *)v11 + 12) )
          {
            v13 = *(_DWORD *)(a1 + 24);
            if ( !v13 )
            {
LABEL_49:
              MEMORY[0] = *v11;
              BUG();
            }
            v15 = *((_BYTE *)v11 + 12);
            v14 = -2;
            LODWORD(v16) = -74;
LABEL_37:
            v16 = (unsigned int)(v16 - 1);
LABEL_14:
            v17 = v13 - 1;
            v18 = *(_QWORD *)(a1 + 8);
            v19 = 1;
            v20 = 0;
            for ( j = v17
                    & (((0xBF58476D1CE4E5B9LL
                       * (v16 | ((unsigned __int64)(((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)) << 32))) >> 31)
                     ^ (484763065 * v16)); ; j = v17 & v24 )
            {
              v22 = (__int64 *)(v18 + 24LL * j);
              v23 = *v22;
              if ( v12 == *v22 && *((_DWORD *)v22 + 2) == v14 && *((_BYTE *)v22 + 12) == v15 )
                break;
              if ( v23 == -4096 )
              {
                if ( *((_DWORD *)v22 + 2) == -1 && *((_BYTE *)v22 + 12) )
                {
                  if ( v20 )
                    v22 = v20;
                  break;
                }
              }
              else if ( v23 == -8192 && *((_DWORD *)v22 + 2) == -2 && *((_BYTE *)v22 + 12) != 1 && !v20 )
              {
                v20 = (__int64 *)(v18 + 24LL * j);
              }
              v24 = v19 + j;
              ++v19;
            }
            *v22 = v12;
            *((_DWORD *)v22 + 2) = *((_DWORD *)v11 + 2);
            *((_BYTE *)v22 + 12) = *((_BYTE *)v11 + 12);
            v22[2] = v11[2];
            ++*(_DWORD *)(a1 + 16);
          }
        }
        v11 += 3;
        if ( v9 == v11 )
        {
          v5 = v27;
          return sub_C7D6A0(v5, v28, 8);
        }
      }
    }
    return sub_C7D6A0(v5, v28, 8);
  }
  else
  {
    v25 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 24 * v25; k != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_DWORD *)(result + 8) = -1;
        *(_BYTE *)(result + 12) = 1;
      }
    }
  }
  return result;
}

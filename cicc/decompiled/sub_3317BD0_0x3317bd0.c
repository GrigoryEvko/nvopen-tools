// Function: sub_3317BD0
// Address: 0x3317bd0
//
__int64 __fastcall sub_3317BD0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r13
  __int64 v5; // r12
  unsigned int v6; // eax
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // r9
  __int64 i; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  int v13; // esi
  int v14; // r8d
  int v15; // r10d
  int v16; // esi
  __int64 v17; // r12
  int v18; // r13d
  __int64 v19; // r14
  unsigned int j; // edx
  __int64 v21; // rdi
  unsigned int v22; // edx
  int v23; // edx
  unsigned __int64 v24; // rdx
  int v25; // r11d
  __int64 v26; // rdx
  __int64 k; // rdx
  __int64 v28; // [rsp+0h] [rbp-40h]
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
  result = sub_C7D670(24LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v29 = 24 * v4;
    v9 = v5 + 24 * v4;
    for ( i = result + 24 * v8; i != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_DWORD *)(result + 8) = -1;
        *(_DWORD *)(result + 16) = 0x7FFFFFFF;
      }
    }
    if ( v9 != v5 )
    {
      v28 = v5;
      v11 = v5;
      while ( 1 )
      {
        v12 = *(_QWORD *)v11;
        if ( *(_QWORD *)v11 )
          goto LABEL_11;
        v23 = *(_DWORD *)(v11 + 8);
        if ( v23 == -1 )
        {
          if ( *(_DWORD *)(v11 + 16) != 0x7FFFFFFF )
            goto LABEL_11;
          v11 += 24;
          if ( v9 == v11 )
            goto LABEL_20;
        }
        else if ( v23 == -2 && *(_DWORD *)(v11 + 16) == 0x80000000 )
        {
          v11 += 24;
          if ( v9 == v11 )
            goto LABEL_20;
        }
        else
        {
LABEL_11:
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = *(_QWORD *)v11;
            MEMORY[8] = *(_DWORD *)(v11 + 8);
            BUG();
          }
          v14 = *(_DWORD *)(v11 + 16);
          v15 = *(_DWORD *)(v11 + 8);
          v16 = v13 - 1;
          v18 = 1;
          v19 = 0;
          for ( j = v16
                  & (((0xBF58476D1CE4E5B9LL
                     * ((unsigned int)(37 * v14)
                      | ((unsigned __int64)(v15 + ((unsigned int)(v12 >> 9) ^ (unsigned int)(v12 >> 4))) << 32))) >> 31)
                   ^ (756364221 * v14)); ; j = v16 & v22 )
          {
            v17 = *(_QWORD *)(a1 + 8);
            v21 = v17 + 24LL * j;
            if ( v12 == *(_QWORD *)v21 && v15 == *(_DWORD *)(v21 + 8) && v14 == *(_DWORD *)(v21 + 16) )
              break;
            if ( !*(_QWORD *)v21 )
            {
              v25 = *(_DWORD *)(v21 + 8);
              if ( v25 == -1 )
              {
                if ( *(_DWORD *)(v21 + 16) == 0x7FFFFFFF )
                {
                  if ( v19 )
                    v21 = v19;
                  break;
                }
              }
              else if ( v25 == -2 && *(_DWORD *)(v21 + 16) == 0x80000000 && !v19 )
              {
                v19 = v17 + 24LL * j;
              }
            }
            v22 = v18 + j;
            ++v18;
          }
          v24 = *(_QWORD *)v11;
          v11 += 24;
          *(_QWORD *)v21 = v24;
          *(_DWORD *)(v21 + 8) = *(_DWORD *)(v11 - 16);
          *(_DWORD *)(v21 + 16) = *(_DWORD *)(v11 - 8);
          ++*(_DWORD *)(a1 + 16);
          if ( v9 == v11 )
          {
LABEL_20:
            v5 = v28;
            return sub_C7D6A0(v5, v29, 8);
          }
        }
      }
    }
    return sub_C7D6A0(v5, v29, 8);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 24 * v26; k != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_DWORD *)(result + 8) = -1;
        *(_DWORD *)(result + 16) = 0x7FFFFFFF;
      }
    }
  }
  return result;
}

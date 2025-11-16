// Function: sub_A46F70
// Address: 0xa46f70
//
__int64 __fastcall sub_A46F70(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r13
  int *v5; // r12
  unsigned int v6; // eax
  __int64 result; // rax
  __int64 v8; // rdx
  int *v9; // r9
  __int64 i; // rdx
  int *v11; // rdx
  int v12; // ecx
  int v13; // r8d
  __int64 v14; // rsi
  int v15; // r8d
  __int64 v16; // rdi
  int *v17; // r14
  int v18; // r11d
  unsigned int j; // eax
  int *v20; // r10
  int v21; // r15d
  __int64 v22; // rax
  unsigned int v23; // eax
  __int64 v24; // rdx
  __int64 k; // rdx

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(int **)(a1 + 8);
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
    v9 = &v5[6 * v4];
    for ( i = result + 24 * v8; i != result; result += 24 )
    {
      if ( result )
      {
        *(_DWORD *)result = -1;
        *(_QWORD *)(result + 8) = -4;
      }
    }
    if ( v9 != v5 )
    {
      v11 = v5;
      while ( 1 )
      {
        v12 = *v11;
        if ( *v11 == -1 )
        {
          if ( *((_QWORD *)v11 + 1) != -4 )
            goto LABEL_12;
          v11 += 6;
          if ( v9 == v11 )
            return sub_C7D6A0(v5, 24 * v4, 8);
        }
        else if ( v12 == -2 && *((_QWORD *)v11 + 1) == -8 )
        {
          v11 += 6;
          if ( v9 == v11 )
            return sub_C7D6A0(v5, 24 * v4, 8);
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
          v14 = *((_QWORD *)v11 + 1);
          v15 = v13 - 1;
          v17 = 0;
          v18 = 1;
          for ( j = v15
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned __int64)(unsigned int)(37 * v12) << 32)
                      | ((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4))) >> 31)
                   ^ (484763065 * (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)))); ; j = v15 & v23 )
          {
            v16 = *(_QWORD *)(a1 + 8);
            v20 = (int *)(v16 + 24LL * j);
            v21 = *v20;
            if ( v12 == *v20 && *((_QWORD *)v20 + 1) == v14 )
              break;
            if ( v21 == -1 )
            {
              if ( *((_QWORD *)v20 + 1) == -4 )
              {
                if ( v17 )
                  v20 = v17;
                break;
              }
            }
            else if ( v21 == -2 && *((_QWORD *)v20 + 1) == -8 && !v17 )
            {
              v17 = (int *)(v16 + 24LL * j);
            }
            v23 = v18 + j;
            ++v18;
          }
          *v20 = v12;
          v22 = *((_QWORD *)v11 + 1);
          v11 += 6;
          *((_QWORD *)v20 + 1) = v22;
          v20[4] = *(v11 - 2);
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
        *(_DWORD *)result = -1;
        *(_QWORD *)(result + 8) = -4;
      }
    }
  }
  return result;
}

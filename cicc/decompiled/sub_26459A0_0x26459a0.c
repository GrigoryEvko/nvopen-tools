// Function: sub_26459A0
// Address: 0x26459a0
//
__int64 __fastcall sub_26459A0(__int64 a1, int a2)
{
  __int64 v3; // r13
  __int64 v4; // r12
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // r9
  __int64 i; // rdx
  __int64 v10; // rdx
  __int64 v11; // rcx
  int v12; // r8d
  int v13; // esi
  __int64 v14; // rdi
  __int64 *v15; // r14
  int v16; // r10d
  int v17; // r11d
  unsigned int j; // eax
  __int64 *v19; // r8
  __int64 v20; // r15
  int v21; // eax
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int64 k; // rdx

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
    v8 = v4 + 24 * v3;
    for ( i = result + 24 * v7; i != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_DWORD *)(result + 8) = -1;
      }
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      while ( 1 )
      {
        v11 = *(_QWORD *)v10;
        if ( *(_QWORD *)v10 == -4096 )
        {
          if ( *(_DWORD *)(v10 + 8) != -1 )
            goto LABEL_12;
          v10 += 24;
          if ( v8 == v10 )
            return sub_C7D6A0(v4, 24 * v3, 8);
        }
        else if ( v11 == -8192 && *(_DWORD *)(v10 + 8) == -2 )
        {
          v10 += 24;
          if ( v8 == v10 )
            return sub_C7D6A0(v4, 24 * v3, 8);
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
          v13 = *(_DWORD *)(v10 + 8);
          v15 = 0;
          v16 = v12 - 1;
          v17 = 1;
          for ( j = (v12 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * ((unsigned int)(37 * v13)
                      | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))) >> 31)
                   ^ (756364221 * v13)); ; j = v16 & v22 )
          {
            v14 = *(_QWORD *)(a1 + 8);
            v19 = (__int64 *)(v14 + 24LL * j);
            v20 = *v19;
            if ( v11 == *v19 && *((_DWORD *)v19 + 2) == v13 )
              break;
            if ( v20 == -4096 )
            {
              if ( *((_DWORD *)v19 + 2) == -1 )
              {
                if ( v15 )
                  v19 = v15;
                break;
              }
            }
            else if ( v20 == -8192 && *((_DWORD *)v19 + 2) == -2 && !v15 )
            {
              v15 = (__int64 *)(v14 + 24LL * j);
            }
            v22 = v17 + j;
            ++v17;
          }
          *v19 = v11;
          v21 = *(_DWORD *)(v10 + 8);
          v10 += 24;
          *((_DWORD *)v19 + 2) = v21;
          *((_DWORD *)v19 + 4) = *(_DWORD *)(v10 - 8);
          ++*(_DWORD *)(a1 + 16);
          if ( v8 == v10 )
            return sub_C7D6A0(v4, 24 * v3, 8);
        }
      }
    }
    return sub_C7D6A0(v4, 24 * v3, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 24 * v23; k != result; result += 24 )
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

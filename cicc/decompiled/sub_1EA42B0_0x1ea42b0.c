// Function: sub_1EA42B0
// Address: 0x1ea42b0
//
_DWORD *__fastcall sub_1EA42B0(__int64 a1, int a2)
{
  __int64 v2; // r13
  int *v4; // r12
  unsigned __int64 v5; // rax
  _DWORD *result; // rax
  int *v7; // r8
  _DWORD *i; // rdx
  int *v9; // rdx
  int v10; // ecx
  int v11; // esi
  int v12; // edi
  int v13; // esi
  __int64 v14; // r9
  int v15; // r14d
  int *v16; // r13
  unsigned __int64 v17; // r10
  unsigned __int64 v18; // r10
  unsigned int j; // eax
  int *v20; // r10
  int v21; // r11d
  int v22; // eax
  unsigned int v23; // eax
  __int64 v24; // rdx
  _DWORD *k; // rdx

  v2 = *(unsigned int *)(a1 + 24);
  v4 = *(int **)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_DWORD *)sub_22077B0(12LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[3 * v2];
    for ( i = &result[3 * *(unsigned int *)(a1 + 24)]; i != result; result += 3 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = -1;
      }
    }
    if ( v7 != v4 )
    {
      v9 = v4;
      while ( 1 )
      {
        v10 = *v9;
        if ( *v9 == -1 )
        {
          if ( v9[1] != -1 )
            goto LABEL_12;
          v9 += 3;
          if ( v7 == v9 )
            return (_DWORD *)j___libc_free_0(v4);
        }
        else if ( v10 == -2 && v9[1] == -2 )
        {
          v9 += 3;
          if ( v7 == v9 )
            return (_DWORD *)j___libc_free_0(v4);
        }
        else
        {
LABEL_12:
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = 0;
            BUG();
          }
          v12 = v9[1];
          v13 = v11 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          v17 = ((((unsigned int)(37 * v12) | ((unsigned __int64)(unsigned int)(37 * v10) << 32))
                - 1
                - ((unsigned __int64)(unsigned int)(37 * v12) << 32)) >> 22)
              ^ (((unsigned int)(37 * v12) | ((unsigned __int64)(unsigned int)(37 * v10) << 32))
               - 1
               - ((unsigned __int64)(unsigned int)(37 * v12) << 32));
          v18 = ((9 * (((v17 - 1 - (v17 << 13)) >> 8) ^ (v17 - 1 - (v17 << 13)))) >> 15)
              ^ (9 * (((v17 - 1 - (v17 << 13)) >> 8) ^ (v17 - 1 - (v17 << 13))));
          for ( j = v13 & (((v18 - 1 - (v18 << 27)) >> 31) ^ (v18 - 1 - ((_DWORD)v18 << 27))); ; j = v13 & v23 )
          {
            v20 = (int *)(v14 + 12LL * j);
            v21 = *v20;
            if ( v10 == *v20 && v20[1] == v12 )
              break;
            if ( v21 == -1 )
            {
              if ( v20[1] == -1 )
              {
                if ( v16 )
                  v20 = v16;
                break;
              }
            }
            else if ( v21 == -2 && v20[1] == -2 && !v16 )
            {
              v16 = (int *)(v14 + 12LL * j);
            }
            v23 = v15 + j;
            ++v15;
          }
          *v20 = v10;
          v22 = v9[1];
          v9 += 3;
          v20[1] = v22;
          v20[2] = *(v9 - 1);
          ++*(_DWORD *)(a1 + 16);
          if ( v7 == v9 )
            return (_DWORD *)j___libc_free_0(v4);
        }
      }
    }
    return (_DWORD *)j___libc_free_0(v4);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[3 * v24]; k != result; result += 3 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = -1;
      }
    }
  }
  return result;
}

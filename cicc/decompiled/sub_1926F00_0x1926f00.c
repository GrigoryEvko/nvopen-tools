// Function: sub_1926F00
// Address: 0x1926f00
//
_DWORD *__fastcall sub_1926F00(__int64 a1, int a2)
{
  __int64 v3; // rbx
  unsigned int *v4; // r12
  unsigned __int64 v5; // rax
  _DWORD *result; // rax
  unsigned int *v7; // rbx
  _DWORD *i; // rdx
  char **j; // r14
  __int64 v10; // rdx
  int v11; // ecx
  unsigned int v12; // esi
  __int64 v13; // rcx
  __int64 v14; // rdi
  int v15; // r10d
  int *v16; // r9
  unsigned __int64 v17; // r8
  unsigned __int64 v18; // r8
  unsigned int k; // eax
  int *v20; // r8
  int v21; // r11d
  unsigned int v22; // eax
  unsigned __int64 v23; // rdi
  unsigned int v24; // eax
  __int64 v25; // rdx
  _DWORD *m; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(unsigned int **)(a1 + 8);
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
  result = (_DWORD *)sub_22077B0(40LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[10 * v3];
    for ( i = &result[10 * *(unsigned int *)(a1 + 24)]; i != result; result += 10 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = -1;
      }
    }
    if ( v7 != v4 )
    {
      for ( j = (char **)v4; v7 != (unsigned int *)j; j += 5 )
      {
        while ( 1 )
        {
          v10 = *(unsigned int *)j;
          if ( (_DWORD)v10 != -1 )
            break;
          if ( *((_DWORD *)j + 1) == -1 )
          {
LABEL_22:
            j += 5;
            if ( v7 == (unsigned int *)j )
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
            v12 = *((_DWORD *)j + 1);
            v13 = (unsigned int)(v11 - 1);
            v15 = 1;
            v16 = 0;
            v17 = ((((37 * v12) | ((unsigned __int64)(unsigned int)(37 * v10) << 32))
                  - 1
                  - ((unsigned __int64)(37 * v12) << 32)) >> 22)
                ^ (((37 * v12) | ((unsigned __int64)(unsigned int)(37 * v10) << 32))
                 - 1
                 - ((unsigned __int64)(37 * v12) << 32));
            v18 = ((9 * (((v17 - 1 - (v17 << 13)) >> 8) ^ (v17 - 1 - (v17 << 13)))) >> 15)
                ^ (9 * (((v17 - 1 - (v17 << 13)) >> 8) ^ (v17 - 1 - (v17 << 13))));
            for ( k = v13 & (((v18 - 1 - (v18 << 27)) >> 31) ^ (v18 - 1 - ((_DWORD)v18 << 27))); ; k = v13 & v24 )
            {
              v14 = *(_QWORD *)(a1 + 8);
              v20 = (int *)(v14 + 40LL * k);
              v21 = *v20;
              if ( *(_QWORD *)v20 == __PAIR64__(v12, v10) )
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
                v16 = (int *)(v14 + 40LL * k);
              }
              v24 = v15 + k;
              ++v15;
            }
            *v20 = v10;
            v22 = *((_DWORD *)j + 1);
            *((_QWORD *)v20 + 2) = 0x200000000LL;
            v20[1] = v22;
            *((_QWORD *)v20 + 1) = v20 + 6;
            if ( *((_DWORD *)j + 4) )
              sub_191FDF0((__int64)(v20 + 2), j + 1, v10, v13, (int)v20, (int)v16);
            ++*(_DWORD *)(a1 + 16);
            v23 = (unsigned __int64)j[1];
            if ( (char **)v23 == j + 3 )
              goto LABEL_22;
            _libc_free(v23);
            j += 5;
            if ( v7 == (unsigned int *)j )
              return (_DWORD *)j___libc_free_0(v4);
          }
        }
        if ( (_DWORD)v10 != -2 || *((_DWORD *)j + 1) != -2 )
          goto LABEL_12;
      }
    }
    return (_DWORD *)j___libc_free_0(v4);
  }
  else
  {
    v25 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = &result[10 * v25]; m != result; result += 10 )
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

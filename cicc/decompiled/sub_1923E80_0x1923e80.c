// Function: sub_1923E80
// Address: 0x1923e80
//
_DWORD *__fastcall sub_1923E80(__int64 a1, int a2)
{
  __int64 v3; // rbx
  unsigned int *v4; // r12
  unsigned __int64 v5; // rax
  _DWORD *result; // rax
  __int64 v7; // rcx
  unsigned int *v8; // rbx
  _DWORD *i; // rdx
  char **j; // r14
  __int64 v11; // rdx
  int v12; // ecx
  unsigned int v13; // esi
  __int64 v14; // rcx
  __int64 v15; // rdi
  int v16; // r10d
  int *v17; // r9
  unsigned __int64 v18; // r8
  unsigned __int64 v19; // r8
  unsigned int k; // eax
  int *v21; // r8
  int v22; // r11d
  unsigned int v23; // eax
  unsigned __int64 v24; // rdi
  unsigned int v25; // eax
  __int64 v26; // rcx
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
  result = (_DWORD *)sub_22077B0(56LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[14 * v3];
    for ( i = &result[14 * v7]; i != result; result += 14 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = -1;
      }
    }
    if ( v8 != v4 )
    {
      for ( j = (char **)v4; v8 != (unsigned int *)j; j += 7 )
      {
        while ( 1 )
        {
          v11 = *(unsigned int *)j;
          if ( (_DWORD)v11 != -1 )
            break;
          if ( *((_DWORD *)j + 1) == -1 )
          {
LABEL_22:
            j += 7;
            if ( v8 == (unsigned int *)j )
              return (_DWORD *)j___libc_free_0(v4);
          }
          else
          {
LABEL_12:
            v12 = *(_DWORD *)(a1 + 24);
            if ( !v12 )
            {
              MEMORY[0] = 0;
              BUG();
            }
            v13 = *((_DWORD *)j + 1);
            v14 = (unsigned int)(v12 - 1);
            v16 = 1;
            v17 = 0;
            v18 = ((((37 * v13) | ((unsigned __int64)(unsigned int)(37 * v11) << 32))
                  - 1
                  - ((unsigned __int64)(37 * v13) << 32)) >> 22)
                ^ (((37 * v13) | ((unsigned __int64)(unsigned int)(37 * v11) << 32))
                 - 1
                 - ((unsigned __int64)(37 * v13) << 32));
            v19 = ((9 * (((v18 - 1 - (v18 << 13)) >> 8) ^ (v18 - 1 - (v18 << 13)))) >> 15)
                ^ (9 * (((v18 - 1 - (v18 << 13)) >> 8) ^ (v18 - 1 - (v18 << 13))));
            for ( k = v14 & (((v19 - 1 - (v19 << 27)) >> 31) ^ (v19 - 1 - ((_DWORD)v19 << 27))); ; k = v14 & v25 )
            {
              v15 = *(_QWORD *)(a1 + 8);
              v21 = (int *)(v15 + 56LL * k);
              v22 = *v21;
              if ( *(_QWORD *)v21 == __PAIR64__(v13, v11) )
                break;
              if ( v22 == -1 )
              {
                if ( v21[1] == -1 )
                {
                  if ( v17 )
                    v21 = v17;
                  break;
                }
              }
              else if ( v22 == -2 && v21[1] == -2 && !v17 )
              {
                v17 = (int *)(v15 + 56LL * k);
              }
              v25 = v16 + k;
              ++v16;
            }
            *v21 = v11;
            v23 = *((_DWORD *)j + 1);
            *((_QWORD *)v21 + 2) = 0x400000000LL;
            v21[1] = v23;
            *((_QWORD *)v21 + 1) = v21 + 6;
            if ( *((_DWORD *)j + 4) )
              sub_191FDF0((__int64)(v21 + 2), j + 1, v11, v14, (int)v21, (int)v17);
            ++*(_DWORD *)(a1 + 16);
            v24 = (unsigned __int64)j[1];
            if ( (char **)v24 == j + 3 )
              goto LABEL_22;
            _libc_free(v24);
            j += 7;
            if ( v8 == (unsigned int *)j )
              return (_DWORD *)j___libc_free_0(v4);
          }
        }
        if ( (_DWORD)v11 != -2 || *((_DWORD *)j + 1) != -2 )
          goto LABEL_12;
      }
    }
    return (_DWORD *)j___libc_free_0(v4);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = &result[14 * v26]; m != result; result += 14 )
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

// Function: sub_1468310
// Address: 0x1468310
//
_QWORD *__fastcall sub_1468310(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r12
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // rbx
  _QWORD *i; // rdx
  char **j; // r14
  char *v11; // rdx
  int v12; // ecx
  char *v13; // rsi
  int v14; // ecx
  __int64 v15; // rdi
  __int64 *v16; // r10
  __int64 v17; // r8
  int v18; // r9d
  unsigned __int64 v19; // r8
  unsigned __int64 v20; // r8
  unsigned int k; // r8d
  __int64 *v22; // rax
  __int64 v23; // r11
  char *v24; // rdx
  unsigned __int64 v25; // rdi
  unsigned int v26; // r8d
  __int64 v27; // rdx
  _QWORD *m; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
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
  result = (_QWORD *)sub_22077B0((unsigned __int64)(unsigned int)v5 << 6);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[8 * v3];
    for ( i = &result[8 * v7]; i != result; result += 8 )
    {
      if ( result )
      {
        *result = -8;
        result[1] = -8;
      }
    }
    if ( v8 != v4 )
    {
      for ( j = (char **)v4; v8 != (__int64 *)j; j += 8 )
      {
        while ( 1 )
        {
          v11 = *j;
          if ( *j != (char *)-8LL )
            break;
          if ( j[1] == (char *)-8LL )
          {
LABEL_22:
            j += 8;
            if ( v8 == (__int64 *)j )
              return (_QWORD *)j___libc_free_0(v4);
          }
          else
          {
LABEL_12:
            v12 = *(_DWORD *)(a1 + 24);
            if ( !v12 )
            {
              MEMORY[0] = *j;
              BUG();
            }
            v13 = j[1];
            v14 = v12 - 1;
            v16 = 0;
            v17 = ((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4);
            v18 = 1;
            v19 = (((v17 | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))
                  - 1
                  - (v17 << 32)) >> 22)
                ^ ((v17 | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))
                 - 1
                 - (v17 << 32));
            v20 = ((9 * (((v19 - 1 - (v19 << 13)) >> 8) ^ (v19 - 1 - (v19 << 13)))) >> 15)
                ^ (9 * (((v19 - 1 - (v19 << 13)) >> 8) ^ (v19 - 1 - (v19 << 13))));
            for ( k = v14 & (((v20 - 1 - (v20 << 27)) >> 31) ^ (v20 - 1 - ((_DWORD)v20 << 27))); ; k = v14 & v26 )
            {
              v15 = *(_QWORD *)(a1 + 8);
              v22 = (__int64 *)(v15 + ((unsigned __int64)k << 6));
              v23 = *v22;
              if ( v11 == (char *)*v22 && (char *)v22[1] == v13 )
                break;
              if ( v23 == -8 )
              {
                if ( v22[1] == -8 )
                {
                  if ( v16 )
                    v22 = v16;
                  break;
                }
              }
              else if ( v23 == -16 && v22[1] == -16 && !v16 )
              {
                v16 = (__int64 *)(v15 + ((unsigned __int64)k << 6));
              }
              v26 = v18 + k;
              ++v18;
            }
            *v22 = (__int64)v11;
            v22[1] = (__int64)j[1];
            v24 = j[2];
            v22[4] = 0x300000000LL;
            v22[2] = (__int64)v24;
            v22[3] = (__int64)(v22 + 5);
            if ( *((_DWORD *)j + 8) )
              sub_14532C0((__int64)(v22 + 3), j + 3);
            ++*(_DWORD *)(a1 + 16);
            v25 = (unsigned __int64)j[3];
            if ( (char **)v25 == j + 5 )
              goto LABEL_22;
            _libc_free(v25);
            j += 8;
            if ( v8 == (__int64 *)j )
              return (_QWORD *)j___libc_free_0(v4);
          }
        }
        if ( v11 != (char *)-16LL || j[1] != (char *)-16LL )
          goto LABEL_12;
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v27 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = &result[8 * v27]; m != result; result += 8 )
    {
      if ( result )
      {
        *result = -8;
        result[1] = -8;
      }
    }
  }
  return result;
}

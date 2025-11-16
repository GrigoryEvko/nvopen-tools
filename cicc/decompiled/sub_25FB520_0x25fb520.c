// Function: sub_25FB520
// Address: 0x25fb520
//
char *__fastcall sub_25FB520(char *dest, char *a2, char *a3)
{
  char *v3; // rbx
  char *result; // rax
  int *v6; // r12
  __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdi
  int *v11; // rdx
  __int64 j; // rcx
  int v13; // r8d
  __int64 v14; // rdx
  int *v15; // r8
  __int64 v16; // rcx
  __int64 i; // rdx
  int v18; // r9d
  __int64 v19; // rax
  int v20; // edx
  int v21; // r13d
  __int64 v22; // rax
  int v23; // r14d
  int *v24; // r13

  v3 = a3;
  if ( dest == a2 )
    return v3;
  result = dest;
  if ( a2 != a3 )
  {
    v6 = (int *)dest;
    v7 = a2 - dest;
    v3 = &dest[a3 - a2];
    v8 = (a3 - dest) >> 2;
    v9 = (a2 - dest) >> 2;
    if ( v9 == v8 - v9 )
    {
      v19 = 0;
      do
      {
        v20 = *(_DWORD *)&dest[v19];
        *(_DWORD *)&dest[v19] = *(_DWORD *)&a2[v19];
        *(_DWORD *)&a2[v19] = v20;
        v19 += 4;
      }
      while ( v19 != v7 );
      return a2;
    }
    else
    {
      while ( 1 )
      {
        v10 = v8 - v9;
        if ( v9 < v8 - v9 )
          break;
LABEL_12:
        v15 = &v6[v8];
        if ( v10 == 1 )
        {
          v21 = *(v15 - 1);
          if ( v6 != v15 - 1 )
            memmove(v6 + 1, v6, 4 * v8 - 4);
          *v6 = v21;
          return v3;
        }
        v6 = &v15[-v10];
        if ( v9 > 0 )
        {
          v16 = 0x3FFFFFFFFFFFFFFFLL;
          for ( i = 0; i != v9; ++i )
          {
            v18 = v6[v16];
            v6[v16] = v15[v16];
            v15[v16--] = v18;
          }
          v6 -= v9;
        }
        v9 = v8 % v10;
        if ( !(v8 % v10) )
          return v3;
        v8 = v10;
      }
      while ( v9 != 1 )
      {
        v11 = &v6[v9];
        if ( v10 > 0 )
        {
          for ( j = 0; j != v10; ++j )
          {
            v13 = v6[j];
            v6[j] = v11[j];
            v11[j] = v13;
          }
          v6 += v10;
        }
        v14 = v8 % v9;
        if ( !(v8 % v9) )
          return v3;
        v8 = v9;
        v9 -= v14;
        v10 = v8 - v9;
        if ( v9 >= v8 - v9 )
          goto LABEL_12;
      }
      v22 = v8;
      v23 = *v6;
      v24 = &v6[v22];
      if ( &v6[v22] != v6 + 1 )
        memmove(v6, v6 + 1, v22 * 4 - 4);
      *(v24 - 1) = v23;
      return v3;
    }
  }
  return result;
}

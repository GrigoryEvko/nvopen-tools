// Function: sub_F02910
// Address: 0xf02910
//
_QWORD *__fastcall sub_F02910(_QWORD *a1, __int64 **a2)
{
  _QWORD *v2; // rdx
  __int64 v4; // rax
  __int64 v5; // r13
  _QWORD *v6; // rax
  unsigned int v7; // ecx
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rsi
  _QWORD *v13; // rax
  __int64 *v14; // rax
  __int64 v15; // rdx

  v2 = 0;
  v4 = *((unsigned int *)a2 + 3);
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  v5 = 2 * v4;
  if ( v4 )
  {
    v6 = (_QWORD *)sub_22077B0(16 * v4);
    v2 = &v6[v5];
    *a1 = v6;
    a1[1] = v6;
    a1[2] = &v6[v5];
    do
    {
      if ( v6 )
      {
        *v6 = 0;
        v6[1] = 0;
      }
      v6 += 2;
    }
    while ( v6 != v2 );
  }
  a1[1] = v2;
  v7 = *((_DWORD *)a2 + 2);
  if ( v7 )
  {
    v8 = **a2;
    v9 = *a2;
    if ( v8 == -8 || !v8 )
    {
      do
      {
        do
        {
          v10 = v9[1];
          ++v9;
        }
        while ( !v10 );
      }
      while ( v10 == -8 );
    }
    v11 = (__int64)&(*a2)[v7];
    if ( (__int64 *)v11 != v9 )
    {
      while ( 1 )
      {
        v12 = *(_QWORD *)*v9;
        v13 = (_QWORD *)(*a1 + 16LL * *(unsigned int *)(*v9 + 8));
        *v13 = *v9 + 16;
        v13[1] = v12;
        v14 = v9 + 1;
        v15 = v9[1];
        if ( v15 != -8 )
          goto LABEL_14;
        do
        {
          do
          {
            v15 = v14[1];
            ++v14;
          }
          while ( v15 == -8 );
LABEL_14:
          ;
        }
        while ( !v15 );
        if ( v14 == (__int64 *)v11 )
          return a1;
        v9 = v14;
      }
    }
  }
  return a1;
}

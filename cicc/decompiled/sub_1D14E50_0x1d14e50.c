// Function: sub_1D14E50
// Address: 0x1d14e50
//
void *__fastcall sub_1D14E50(void **a1, void **a2)
{
  void *result; // rax
  void *v4; // r14
  void *v5; // rbx
  void *v6; // rax
  _QWORD *v7; // r15
  __int64 v8; // rsi
  _QWORD *i; // r14
  _QWORD *v10; // r15
  __int64 v11; // rsi
  _QWORD *v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // [rsp+0h] [rbp-40h]
  __int64 v17; // [rsp+8h] [rbp-38h]

  result = sub_16982C0();
  v4 = *a2;
  v5 = result;
  if ( *a1 == result )
  {
    if ( result == v4 )
    {
      if ( a2 != a1 )
      {
        v10 = a1[1];
        if ( v10 )
        {
          v11 = 4LL * *(v10 - 1);
          v12 = &v10[v11];
          while ( v10 != v12 )
          {
            v12 -= 4;
            if ( (void *)v12[1] == v4 )
            {
              v13 = v12[2];
              if ( v13 )
              {
                v14 = 32LL * *(_QWORD *)(v13 - 8);
                v15 = v13 + v14;
                if ( v13 != v13 + v14 )
                {
                  do
                  {
                    v16 = v13;
                    v17 = v15 - 32;
                    sub_127D120((_QWORD *)(v15 - 24));
                    v15 = v17;
                    v13 = v16;
                  }
                  while ( v16 != v17 );
                }
                j_j_j___libc_free_0_0(v13 - 8);
              }
            }
            else
            {
              sub_1698460((__int64)(v12 + 1));
            }
          }
          j_j_j___libc_free_0_0(v10 - 1);
        }
        return sub_169C7E0(a1, a2);
      }
    }
    else if ( a1 != a2 )
    {
      v7 = a1[1];
      if ( !v7 )
        return (void *)sub_1698450((__int64)a1, (__int64)a2);
      v8 = 4LL * *(v7 - 1);
      for ( i = &v7[v8]; v7 != i; sub_127D120(i + 1) )
        i -= 4;
      j_j_j___libc_free_0_0(v7 - 1);
      v6 = *a2;
LABEL_6:
      if ( v5 != v6 )
        return (void *)sub_1698450((__int64)a1, (__int64)a2);
      return sub_169C7E0(a1, a2);
    }
  }
  else
  {
    if ( result != v4 )
      return (void *)sub_16983E0((__int64)a1, (__int64)a2);
    if ( a1 != a2 )
    {
      sub_1698460((__int64)a1);
      v6 = *a2;
      goto LABEL_6;
    }
  }
  return result;
}

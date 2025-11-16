// Function: sub_1F78E70
// Address: 0x1f78e70
//
void *__fastcall sub_1F78E70(void **a1, void **a2)
{
  void *result; // rax
  void *v5; // r13
  void *v6; // rbx
  void *v7; // rax
  _QWORD *v8; // r13
  __int64 v9; // rsi
  _QWORD *i; // r12
  _QWORD *v11; // rax
  _QWORD *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // [rsp+0h] [rbp-60h]
  __int64 v21; // [rsp+8h] [rbp-58h]
  __int64 v22; // [rsp+18h] [rbp-48h]
  __int64 v23; // [rsp+20h] [rbp-40h]
  __int64 v24; // [rsp+20h] [rbp-40h]
  __int64 v25; // [rsp+20h] [rbp-40h]
  _QWORD *v26; // [rsp+28h] [rbp-38h]

  result = sub_16982C0();
  v5 = *a2;
  v6 = result;
  if ( *a1 == result )
  {
    if ( result == v5 )
    {
      if ( a2 != a1 )
      {
        v11 = a1[1];
        v26 = v11;
        if ( v11 )
        {
          v12 = &v11[4 * *(v11 - 1)];
          if ( v11 != v12 )
          {
            do
            {
              v12 -= 4;
              if ( (void *)v12[1] == v5 )
              {
                v13 = v12[2];
                v21 = v13;
                if ( v13 )
                {
                  v14 = v13 + 32LL * *(_QWORD *)(v13 - 8);
                  if ( v13 != v14 )
                  {
                    do
                    {
                      v14 -= 32;
                      if ( *(void **)(v14 + 8) == v5 )
                      {
                        v15 = *(_QWORD *)(v14 + 16);
                        v20 = v15;
                        if ( v15 )
                        {
                          v16 = v15 + 32LL * *(_QWORD *)(v15 - 8);
                          if ( v15 != v16 )
                          {
                            do
                            {
                              v16 -= 32;
                              if ( *(void **)(v16 + 8) == v5 )
                              {
                                v17 = *(_QWORD *)(v16 + 16);
                                if ( v17 )
                                {
                                  v18 = 32LL * *(_QWORD *)(v17 - 8);
                                  v19 = v17 + v18;
                                  if ( v17 != v17 + v18 )
                                  {
                                    do
                                    {
                                      v22 = v16;
                                      v24 = v19 - 32;
                                      sub_127D120((_QWORD *)(v19 - 24));
                                      v19 = v24;
                                      v16 = v22;
                                    }
                                    while ( v17 != v24 );
                                  }
                                  v25 = v16;
                                  j_j_j___libc_free_0_0(v17 - 8);
                                  v16 = v25;
                                }
                              }
                              else
                              {
                                v23 = v16;
                                sub_1698460(v16 + 8);
                                v16 = v23;
                              }
                            }
                            while ( v20 != v16 );
                          }
                          j_j_j___libc_free_0_0(v20 - 8);
                        }
                      }
                      else
                      {
                        sub_1698460(v14 + 8);
                      }
                    }
                    while ( v21 != v14 );
                  }
                  j_j_j___libc_free_0_0(v21 - 8);
                }
              }
              else
              {
                sub_1698460((__int64)(v12 + 1));
              }
            }
            while ( v26 != v12 );
          }
          j_j_j___libc_free_0_0(v26 - 1);
        }
        return sub_169C7E0(a1, a2);
      }
    }
    else if ( a1 != a2 )
    {
      v8 = a1[1];
      if ( !v8 )
        return (void *)sub_1698450((__int64)a1, (__int64)a2);
      v9 = 4LL * *(v8 - 1);
      for ( i = &v8[v9]; v8 != i; sub_127D120(i + 1) )
        i -= 4;
      j_j_j___libc_free_0_0(v8 - 1);
      v7 = *a2;
LABEL_6:
      if ( v6 != v7 )
        return (void *)sub_1698450((__int64)a1, (__int64)a2);
      return sub_169C7E0(a1, a2);
    }
  }
  else
  {
    if ( result != v5 )
      return (void *)sub_16983E0((__int64)a1, (__int64)a2);
    if ( a1 != a2 )
    {
      sub_1698460((__int64)a1);
      v7 = *a2;
      goto LABEL_6;
    }
  }
  return result;
}

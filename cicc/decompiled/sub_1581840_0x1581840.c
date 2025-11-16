// Function: sub_1581840
// Address: 0x1581840
//
__int64 __fastcall sub_1581840(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // r14
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rsi
  __int64 i; // r14
  __int64 v12; // r15
  __int64 v13; // rsi
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // [rsp+0h] [rbp-40h]
  __int64 v19; // [rsp+8h] [rbp-38h]

  result = sub_16982C0(a1, a2, a3, a4);
  v6 = *a2;
  v7 = result;
  if ( *a1 == result )
  {
    if ( result == v6 )
    {
      if ( a2 != a1 )
      {
        v12 = a1[1];
        if ( v12 )
        {
          v13 = 32LL * *(_QWORD *)(v12 - 8);
          v14 = v12 + v13;
          while ( v12 != v14 )
          {
            v14 -= 32;
            if ( *(_QWORD *)(v14 + 8) == v6 )
            {
              v15 = *(_QWORD *)(v14 + 16);
              if ( v15 )
              {
                v16 = 32LL * *(_QWORD *)(v15 - 8);
                v17 = v15 + v16;
                if ( v15 != v15 + v16 )
                {
                  do
                  {
                    v18 = v15;
                    v19 = v17 - 32;
                    sub_127D120((_QWORD *)(v17 - 24));
                    v17 = v19;
                    v15 = v18;
                  }
                  while ( v18 != v19 );
                }
                j_j_j___libc_free_0_0(v15 - 8);
              }
            }
            else
            {
              sub_1698460(v14 + 8);
            }
          }
          j_j_j___libc_free_0_0(v12 - 8);
        }
        return sub_169C7E0(a1, a2);
      }
    }
    else if ( a1 != a2 )
    {
      v9 = a1[1];
      if ( !v9 )
        return sub_1698450(a1, a2);
      v10 = 32LL * *(_QWORD *)(v9 - 8);
      for ( i = v9 + v10; v9 != i; sub_127D120((_QWORD *)(i + 8)) )
        i -= 32;
      j_j_j___libc_free_0_0(v9 - 8);
      v8 = *a2;
LABEL_6:
      if ( v7 != v8 )
        return sub_1698450(a1, a2);
      return sub_169C7E0(a1, a2);
    }
  }
  else
  {
    if ( result != v6 )
      return sub_16983E0(a1, a2);
    if ( a1 != a2 )
    {
      sub_1698460(a1);
      v8 = *a2;
      goto LABEL_6;
    }
  }
  return result;
}

// Function: sub_127D120
// Address: 0x127d120
//
__int64 __fastcall sub_127D120(_QWORD *a1)
{
  __int64 result; // rax
  __int64 v2; // r14
  __int64 v3; // rbx
  __int64 v4; // rsi
  __int64 v5; // r12
  __int64 v6; // r15
  __int64 v7; // rsi
  __int64 v8; // r13
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // [rsp+0h] [rbp-40h]
  __int64 v13; // [rsp+8h] [rbp-38h]

  result = sub_16982C0();
  if ( *a1 != result )
    return sub_1698460(a1);
  v2 = a1[1];
  if ( v2 )
  {
    v3 = result;
    v4 = 32LL * *(_QWORD *)(v2 - 8);
    v5 = v2 + v4;
    while ( v2 != v5 )
    {
      v5 -= 32;
      if ( v3 == *(_QWORD *)(v5 + 8) )
      {
        v6 = *(_QWORD *)(v5 + 16);
        if ( v6 )
        {
          v7 = 32LL * *(_QWORD *)(v6 - 8);
          v8 = v6 + v7;
          while ( v6 != v8 )
          {
            v8 -= 32;
            if ( v3 == *(_QWORD *)(v8 + 8) )
            {
              v9 = *(_QWORD *)(v8 + 16);
              if ( v9 )
              {
                v10 = 32LL * *(_QWORD *)(v9 - 8);
                v11 = v9 + v10;
                if ( v9 != v9 + v10 )
                {
                  do
                  {
                    v12 = v9;
                    v13 = v11 - 32;
                    sub_127D120(v11 - 24);
                    v11 = v13;
                    v9 = v12;
                  }
                  while ( v12 != v13 );
                }
                j_j_j___libc_free_0_0(v9 - 8);
              }
            }
            else
            {
              sub_1698460(v8 + 8);
            }
          }
          j_j_j___libc_free_0_0(v6 - 8);
        }
      }
      else
      {
        sub_1698460(v5 + 8);
      }
    }
    return j_j_j___libc_free_0_0(v2 - 8);
  }
  return result;
}

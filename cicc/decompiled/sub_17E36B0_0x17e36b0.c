// Function: sub_17E36B0
// Address: 0x17e36b0
//
void __fastcall sub_17E36B0(char *a1, char *a2, __int64 *a3)
{
  __int64 v3; // rbx
  __int64 *v4; // r14
  char *v5; // r12
  char *v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 *v9; // r8
  __int64 *v10; // r14
  __int64 v11; // r15
  __int64 v12; // rbx
  __int64 *v13; // rdi
  __int64 v14; // rax
  __int64 *v15; // r8
  __int64 *v16; // r14
  __int64 v17; // r12
  __int64 *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-60h]
  __int64 v22; // [rsp+0h] [rbp-60h]
  __int64 *v23; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+28h] [rbp-38h]

  v3 = a2 - a1;
  v4 = (__int64 *)((char *)a3 + a2 - a1);
  v26 = (a2 - a1) >> 3;
  if ( a2 - a1 <= 48 )
  {
    sub_17E26D0(a1, a2);
  }
  else
  {
    v5 = a1;
    do
    {
      v6 = v5;
      v5 += 56;
      sub_17E26D0(v6, v5);
    }
    while ( a2 - v5 > 48 );
    sub_17E26D0(v5, a2);
    if ( v3 > 56 )
    {
      v23 = v4;
      v7 = 7;
      while ( 1 )
      {
        v8 = 2 * v7;
        if ( v26 < 2 * v7 )
        {
          v9 = a3;
          v14 = v26;
          v10 = (__int64 *)a1;
        }
        else
        {
          v9 = a3;
          v10 = (__int64 *)a1;
          v21 = v7;
          v11 = 16 * v7;
          v12 = -1 * v7;
          do
          {
            v13 = v10;
            v10 = (__int64 *)((char *)v10 + v11);
            v9 = (__int64 *)sub_17E3550(v13, &v10[v12], &v10[v12], v10, v9);
            v14 = (a2 - (char *)v10) >> 3;
          }
          while ( v8 <= v14 );
          v7 = v21;
        }
        if ( v7 <= v14 )
          v14 = v7;
        v7 *= 4;
        sub_17E3550(v10, &v10[v14], &v10[v14], (__int64 *)a2, v9);
        v15 = (__int64 *)a1;
        if ( v26 < v7 )
          break;
        v22 = v8;
        v16 = a3;
        v17 = 8 * v8 - 8 * v7;
        do
        {
          v18 = v16;
          v16 += v7;
          v15 = sub_17E3070(v18, (__int64 *)((char *)v16 + v17), (__int64 *)((char *)v16 + v17), v16, v15);
          v19 = v23 - v16;
        }
        while ( v7 <= v19 );
        if ( v19 > v22 )
          v19 = v22;
        sub_17E3070(v16, &v16[v19], &v16[v19], v23, v15);
        if ( v26 <= v7 )
          return;
      }
      v20 = v8;
      if ( v26 <= v8 )
        v20 = v26;
      sub_17E3070(a3, &a3[v20], &a3[v20], v23, (__int64 *)a1);
    }
  }
}

// Function: sub_24268C0
// Address: 0x24268c0
//
void __fastcall sub_24268C0(unsigned __int64 *a1, unsigned __int64 *a2, unsigned __int64 *a3)
{
  __int64 v3; // rbx
  unsigned __int64 *v4; // r14
  unsigned __int64 *v5; // r12
  unsigned __int64 *v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // r12
  unsigned __int64 *v9; // r8
  unsigned __int64 *v10; // r14
  __int64 v11; // r15
  __int64 v12; // rbx
  unsigned __int64 *v13; // rdi
  __int64 v14; // rax
  unsigned __int64 *v15; // r8
  unsigned __int64 *v16; // r14
  __int64 v17; // r12
  unsigned __int64 *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-60h]
  __int64 v22; // [rsp+0h] [rbp-60h]
  unsigned __int64 *v23; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+28h] [rbp-38h]

  v3 = (char *)a2 - (char *)a1;
  v4 = (unsigned __int64 *)((char *)a3 + (char *)a2 - (char *)a1);
  v26 = a2 - a1;
  if ( (char *)a2 - (char *)a1 <= 48 )
  {
    sub_2425AC0(a1, a2);
  }
  else
  {
    v5 = a1;
    do
    {
      v6 = v5;
      v5 += 7;
      sub_2425AC0(v6, v5);
    }
    while ( (char *)a2 - (char *)v5 > 48 );
    sub_2425AC0(v5, a2);
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
          v10 = a1;
        }
        else
        {
          v9 = a3;
          v10 = a1;
          v21 = v7;
          v11 = 16 * v7;
          v12 = -1 * v7;
          do
          {
            v13 = v10;
            v10 = (unsigned __int64 *)((char *)v10 + v11);
            v9 = sub_2426760(v13, &v10[v12], &v10[v12], v10, v9);
            v14 = a2 - v10;
          }
          while ( v8 <= v14 );
          v7 = v21;
        }
        if ( v7 <= v14 )
          v14 = v7;
        v7 *= 4;
        sub_2426760(v10, &v10[v14], &v10[v14], a2, v9);
        v15 = a1;
        if ( v26 < v7 )
          break;
        v22 = v8;
        v16 = a3;
        v17 = 8 * v8 - 8 * v7;
        do
        {
          v18 = v16;
          v16 += v7;
          v15 = sub_24265F0(
                  v18,
                  (unsigned __int64 *)((char *)v16 + v17),
                  (unsigned __int64 *)((char *)v16 + v17),
                  v16,
                  v15);
          v19 = v23 - v16;
        }
        while ( v7 <= v19 );
        if ( v19 > v22 )
          v19 = v22;
        sub_24265F0(v16, &v16[v19], &v16[v19], v23, v15);
        if ( v26 <= v7 )
          return;
      }
      v20 = v8;
      if ( v26 <= v8 )
        v20 = v26;
      sub_24265F0(a3, &a3[v20], &a3[v20], v23, a1);
    }
  }
}

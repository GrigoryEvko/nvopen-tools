// Function: sub_39BB490
// Address: 0x39bb490
//
void __fastcall sub_39BB490(__int64 *a1, __int64 *a2, char *a3)
{
  __int64 v3; // rbx
  char *v4; // r14
  __int64 *v5; // r12
  __int64 *v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // r12
  char *v9; // r8
  char *v10; // r14
  __int64 v11; // r15
  __int64 v12; // rbx
  char *v13; // rdi
  __int64 v14; // rax
  char *v15; // r8
  char *v16; // r14
  __int64 v17; // r12
  char *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-60h]
  __int64 v22; // [rsp+0h] [rbp-60h]
  char *v23; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+28h] [rbp-38h]

  v3 = (char *)a2 - (char *)a1;
  v4 = &a3[(char *)a2 - (char *)a1];
  v26 = a2 - a1;
  if ( (char *)a2 - (char *)a1 <= 48 )
  {
    sub_39BB3B0(a1, a2);
  }
  else
  {
    v5 = a1;
    do
    {
      v6 = v5;
      v5 += 7;
      sub_39BB3B0(v6, v5);
    }
    while ( (char *)a2 - (char *)v5 > 48 );
    sub_39BB3B0(v5, a2);
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
          v10 = (char *)a1;
        }
        else
        {
          v9 = a3;
          v10 = (char *)a1;
          v21 = v7;
          v11 = 16 * v7;
          v12 = -8 * v7;
          do
          {
            v13 = v10;
            v10 += v11;
            v9 = sub_39BB200(v13, &v10[v12], &v10[v12], v10, v9);
            v14 = ((char *)a2 - v10) >> 3;
          }
          while ( v8 <= v14 );
          v7 = v21;
        }
        if ( v7 <= v14 )
          v14 = v7;
        v7 *= 4;
        sub_39BB200(v10, &v10[8 * v14], &v10[8 * v14], (char *)a2, v9);
        v15 = (char *)a1;
        if ( v26 < v7 )
          break;
        v22 = v8;
        v16 = a3;
        v17 = 8 * v8 - 8 * v7;
        do
        {
          v18 = v16;
          v16 += 8 * v7;
          v15 = sub_39BB2D0(v18, &v16[v17], &v16[v17], v16, v15);
          v19 = (v23 - v16) >> 3;
        }
        while ( v7 <= v19 );
        if ( v19 > v22 )
          v19 = v22;
        sub_39BB2D0(v16, &v16[8 * v19], &v16[8 * v19], v23, v15);
        if ( v26 <= v7 )
          return;
      }
      v20 = v8;
      if ( v26 <= v8 )
        v20 = v26;
      sub_39BB2D0(a3, &a3[8 * v20], &a3[8 * v20], v23, a1);
    }
  }
}

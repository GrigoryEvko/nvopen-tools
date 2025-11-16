// Function: sub_39BAE30
// Address: 0x39bae30
//
void __fastcall sub_39BAE30(char *src, char *a2, char *a3)
{
  __int64 v3; // rbx
  char *v4; // r14
  char *v5; // r12
  char *v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // r12
  _QWORD *v9; // r8
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

  v3 = a2 - src;
  v4 = &a3[a2 - src];
  v26 = (a2 - src) >> 3;
  if ( a2 - src <= 48 )
  {
    sub_39BAD70(src, a2);
  }
  else
  {
    v5 = src;
    do
    {
      v6 = v5;
      v5 += 56;
      sub_39BAD70(v6, v5);
    }
    while ( a2 - v5 > 48 );
    sub_39BAD70(v5, a2);
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
          v10 = src;
        }
        else
        {
          v9 = a3;
          v10 = src;
          v21 = v7;
          v11 = 16 * v7;
          v12 = -8 * v7;
          do
          {
            v13 = v10;
            v10 += v11;
            v9 = (_QWORD *)sub_39BA920(v13, &v10[v12], &v10[v12], v10, v9);
            v14 = (a2 - v10) >> 3;
          }
          while ( v8 <= v14 );
          v7 = v21;
        }
        if ( v7 <= v14 )
          v14 = v7;
        v7 *= 4;
        sub_39BA920(v10, &v10[8 * v14], &v10[8 * v14], a2, v9);
        v15 = src;
        if ( v26 < v7 )
          break;
        v22 = v8;
        v16 = a3;
        v17 = 8 * v8 - 8 * v7;
        do
        {
          v18 = v16;
          v16 += 8 * v7;
          v15 = sub_39BAA60(v18, &v16[v17], &v16[v17], v16, v15);
          v19 = (v23 - v16) >> 3;
        }
        while ( v7 <= v19 );
        if ( v19 > v22 )
          v19 = v22;
        sub_39BAA60(v16, &v16[8 * v19], &v16[8 * v19], v23, v15);
        if ( v26 <= v7 )
          return;
      }
      v20 = v8;
      if ( v26 <= v8 )
        v20 = v26;
      sub_39BAA60(a3, &a3[8 * v20], &a3[8 * v20], v23, src);
    }
  }
}

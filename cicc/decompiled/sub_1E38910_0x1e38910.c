// Function: sub_1E38910
// Address: 0x1e38910
//
void __fastcall sub_1E38910(char *a1, unsigned __int64 *a2, char *a3)
{
  __int64 v3; // rbx
  char *v4; // r14
  unsigned __int64 *v5; // r12
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // r12
  char *v9; // r8
  __int64 v10; // r15
  char *v11; // r14
  __int64 v12; // rbx
  char *v13; // rdi
  __int64 v14; // rax
  _QWORD *v15; // r8
  char *v16; // r14
  __int64 v17; // r12
  char *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rsi
  __int64 v22; // [rsp+0h] [rbp-60h]
  __int64 v23; // [rsp+0h] [rbp-60h]
  char *v24; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+28h] [rbp-38h]

  v3 = (char *)a2 - a1;
  v4 = &a3[(char *)a2 - a1];
  v28 = ((char *)a2 - a1) >> 4;
  if ( (char *)a2 - a1 <= 96 )
  {
    sub_1E38550((__int64)a1, a2);
  }
  else
  {
    v5 = (unsigned __int64 *)a1;
    do
    {
      v6 = (__int64)v5;
      v5 += 14;
      sub_1E38550(v6, v5);
    }
    while ( (char *)a2 - (char *)v5 > 96 );
    sub_1E38550((__int64)v5, a2);
    if ( v3 > 112 )
    {
      v24 = v4;
      v7 = 7;
      while ( 1 )
      {
        v8 = 2 * v7;
        if ( v28 < 2 * v7 )
        {
          v9 = a3;
          v14 = v28;
          v11 = a1;
        }
        else
        {
          v22 = v7;
          v9 = a3;
          v10 = 32 * v7;
          v11 = a1;
          v12 = -16 * v7;
          do
          {
            v13 = v11;
            v11 += v10;
            v9 = sub_1E37D40(v13, &v11[v12], &v11[v12], v11, v9);
            v14 = ((char *)a2 - v11) >> 4;
          }
          while ( v8 <= v14 );
          v7 = v22;
        }
        if ( v7 <= v14 )
          v14 = v7;
        v7 *= 4;
        sub_1E37D40(v11, &v11[16 * v14], &v11[16 * v14], (char *)a2, v9);
        v15 = a1;
        if ( v28 < v7 )
          break;
        v23 = v8;
        v16 = a3;
        v17 = 16 * v8 - 16 * v7;
        do
        {
          v18 = v16;
          v16 += 16 * v7;
          v15 = sub_1E38140(v18, &v16[v17], &v16[v17], v16, v15);
          v19 = (v24 - v16) >> 4;
        }
        while ( v7 <= v19 );
        v20 = v23;
        if ( v19 <= v23 )
          v20 = (v24 - v16) >> 4;
        sub_1E38140(v16, &v16[16 * v20], &v16[16 * v20], v24, v15);
        if ( v28 <= v7 )
          return;
      }
      v21 = v8;
      if ( v28 <= v8 )
        v21 = v28;
      sub_1E38140(a3, &a3[16 * v21], &a3[16 * v21], v24, a1);
    }
  }
}

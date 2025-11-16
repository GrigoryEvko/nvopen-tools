// Function: sub_150AF00
// Address: 0x150af00
//
char **__fastcall sub_150AF00(char **a1, __int64 a2, int *a3, unsigned __int64 a4)
{
  __int64 v5; // rbx
  __int64 v8; // rax
  char *v9; // rdi
  char *v10; // rsi
  _QWORD *v11; // r13
  char *v12; // rax
  _QWORD *v13; // rdx
  char *v14; // r13
  int *v15; // rbx
  char *v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rdx
  _QWORD v20[8]; // [rsp+0h] [rbp-40h] BYREF

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( a4 > 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::reserve");
  if ( a4 )
  {
    v5 = 2 * a4;
    v8 = sub_22077B0(8 * a4);
    v9 = *a1;
    v10 = a1[1];
    v11 = (_QWORD *)v8;
    v12 = *a1;
    if ( v10 != *a1 )
    {
      v13 = v11;
      do
      {
        if ( v13 )
          *v13 = *(_QWORD *)v12;
        v12 += 8;
        ++v13;
      }
      while ( v10 != v12 );
    }
    if ( v9 )
      j_j___libc_free_0(v9, a1[2] - v9);
    *a1 = (char *)v11;
    a1[1] = (char *)v11;
    v14 = (char *)&v11[(unsigned __int64)v5 / 2];
    v15 = &a3[v5];
    a1[2] = v14;
    do
    {
      while ( 1 )
      {
        v17 = sub_150AAB0(a2, *a3);
        v16 = a1[1];
        v20[0] = v17;
        v20[1] = v18;
        if ( v16 != a1[2] )
          break;
        a3 += 2;
        sub_14F4870(a1, v16, v20);
        if ( v15 == a3 )
          return a1;
      }
      if ( v16 )
      {
        *(_QWORD *)v16 = v20[0];
        v16 = a1[1];
      }
      a3 += 2;
      a1[1] = v16 + 8;
    }
    while ( v15 != a3 );
  }
  return a1;
}

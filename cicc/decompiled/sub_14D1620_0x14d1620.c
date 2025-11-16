// Function: sub_14D1620
// Address: 0x14d1620
//
double __fastcall sub_14D1620(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r14
  char v6; // r13
  __int64 v7; // rax
  __int64 v8; // r12
  _QWORD *v9; // rsi
  __int64 v10; // rsi
  __int64 *v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  double result; // xmm0_8
  __int64 v15; // r12
  __int64 v16; // rsi
  __int64 v17; // rbx
  char v18; // [rsp+1Fh] [rbp-41h] BYREF
  _BYTE v19[8]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v20; // [rsp+28h] [rbp-38h] BYREF
  __int64 v21; // [rsp+30h] [rbp-30h]

  v5 = a1[4];
  v6 = *(_BYTE *)(*a1 + 8LL);
  v7 = sub_16982C0(a1, a2, a3, a4);
  v8 = v7;
  if ( v6 == 2 )
  {
    v13 = (__int64)(a1 + 4);
    if ( v5 == v7 )
      v13 = a1[5] + 8LL;
    return sub_169D890(v13);
  }
  else if ( v6 == 3 )
  {
    v12 = (__int64)(a1 + 4);
    if ( v5 == v7 )
      v12 = a1[5] + 8LL;
    sub_169D8E0(v12);
  }
  else
  {
    v9 = a1 + 4;
    if ( v5 == v7 )
      sub_169C6E0(&v20, v9);
    else
      sub_16986C0(&v20, v9);
    v10 = sub_1698280(&v20);
    sub_16A3360(v19, v10, 0, &v18);
    v11 = &v20;
    if ( v20 == v8 )
      v11 = (__int64 *)(v21 + 8);
    sub_169D8E0(v11);
    if ( v20 == v8 )
    {
      v15 = v21;
      if ( v21 )
      {
        v16 = 32LL * *(_QWORD *)(v21 - 8);
        v17 = v21 + v16;
        if ( v21 != v21 + v16 )
        {
          do
          {
            v17 -= 32;
            sub_127D120((_QWORD *)(v17 + 8));
          }
          while ( v15 != v17 );
        }
        j_j_j___libc_free_0_0(v15 - 8);
      }
    }
    else
    {
      sub_1698460(&v20);
    }
  }
  return result;
}

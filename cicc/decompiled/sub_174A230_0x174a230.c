// Function: sub_174A230
// Address: 0x174a230
//
_BOOL8 __fastcall sub_174A230(__int64 a1, __int16 *a2)
{
  __int64 *v3; // rsi
  void *v4; // rbx
  _BOOL4 v5; // r12d
  __int64 v7; // r13
  __int64 v8; // rsi
  __int64 v9; // rbx
  bool v10; // [rsp+Fh] [rbp-41h] BYREF
  _BYTE v11[8]; // [rsp+10h] [rbp-40h] BYREF
  void *v12; // [rsp+18h] [rbp-38h] BYREF
  __int64 v13; // [rsp+20h] [rbp-30h]

  v3 = (__int64 *)(a1 + 32);
  v4 = sub_16982C0();
  if ( *(void **)(a1 + 32) == v4 )
    sub_169C6E0(&v12, (__int64)v3);
  else
    sub_16986C0(&v12, v3);
  sub_16A3360((__int64)v11, a2, 0, &v10);
  v5 = !v10;
  if ( v12 != v4 )
  {
    sub_1698460((__int64)&v12);
    return v5;
  }
  v7 = v13;
  if ( !v13 )
    return v5;
  v8 = 32LL * *(_QWORD *)(v13 - 8);
  v9 = v13 + v8;
  if ( v13 != v13 + v8 )
  {
    do
    {
      v9 -= 32;
      sub_127D120((_QWORD *)(v9 + 8));
    }
    while ( v7 != v9 );
  }
  j_j_j___libc_free_0_0(v7 - 8);
  return v5;
}

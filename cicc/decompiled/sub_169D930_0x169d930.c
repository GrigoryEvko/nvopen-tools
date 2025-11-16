// Function: sub_169D930
// Address: 0x169d930
//
__int64 __fastcall sub_169D930(__int64 a1, __int64 a2)
{
  __int64 *v3; // rsi
  void *v4; // rbx
  __int64 *v5; // rsi
  _QWORD *v7; // [rsp+0h] [rbp-40h] BYREF
  _QWORD *v8; // [rsp+8h] [rbp-38h]
  _QWORD *v9; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v10; // [rsp+18h] [rbp-28h]

  v3 = (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL);
  v4 = sub_16982C0();
  if ( (void *)*v3 == v4 )
    sub_169D930(&v9, v3);
  else
    sub_169D7E0((__int64)&v9, v3);
  if ( v10 <= 0x40 )
  {
    v7 = v9;
  }
  else
  {
    v7 = (_QWORD *)*v9;
    j_j___libc_free_0_0(v9);
  }
  v5 = (__int64 *)(*(_QWORD *)(a2 + 8) + 40LL);
  if ( v4 == (void *)*v5 )
    sub_169D930(&v9, v5);
  else
    sub_169D7E0((__int64)&v9, v5);
  if ( v10 <= 0x40 )
  {
    v8 = v9;
  }
  else
  {
    v8 = (_QWORD *)*v9;
    j_j___libc_free_0_0(v9);
  }
  sub_16A5110(a1, 128, 2, &v7);
  return a1;
}

// Function: sub_1969510
// Address: 0x1969510
//
__int64 __fastcall sub_1969510(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, _QWORD *a5, __m128i a6, __m128i a7)
{
  __int64 v10; // r13
  __int64 *v12[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v13[8]; // [rsp+10h] [rbp-40h] BYREF

  v10 = sub_1483B20(a5, a2, a3, a6, a7);
  if ( a4 != 1 )
  {
    v13[1] = sub_145CF80((__int64)a5, a3, a4, 0);
    v13[0] = v10;
    v12[0] = v13;
    v12[1] = (__int64 *)0x200000002LL;
    v10 = sub_147EE30(a5, v12, 2u, 0, a6, a7);
    if ( v12[0] != v13 )
      _libc_free((unsigned __int64)v12[0]);
  }
  return sub_14806B0((__int64)a5, a1, v10, 0, 0);
}

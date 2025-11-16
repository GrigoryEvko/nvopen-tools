// Function: sub_14819D0
// Address: 0x14819d0
//
__int64 __fastcall sub_14819D0(_QWORD *a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 v5; // r12
  __int64 *v7[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v8[4]; // [rsp+10h] [rbp-20h] BYREF

  v7[0] = v8;
  v8[0] = a2;
  v8[1] = a3;
  v7[1] = (__int64 *)0x200000002LL;
  v5 = sub_14813B0(a1, v7, a4, a5);
  if ( v7[0] != v8 )
    _libc_free((unsigned __int64)v7[0]);
  return v5;
}

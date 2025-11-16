// Function: sub_1480950
// Address: 0x1480950
//
__int64 __fastcall sub_1480950(_QWORD *a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 v5; // r12
  unsigned __int64 v7[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v8[4]; // [rsp+10h] [rbp-20h] BYREF

  v7[0] = (unsigned __int64)v8;
  v8[0] = a2;
  v8[1] = a3;
  v7[1] = 0x200000002LL;
  v5 = sub_1480880(a1, (__int64)v7, a4, a5);
  if ( (_QWORD *)v7[0] != v8 )
    _libc_free(v7[0]);
  return v5;
}

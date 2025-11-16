// Function: sub_1494E10
// Address: 0x1494e10
//
__int64 *__fastcall sub_1494E10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, __m128i a6)
{
  __int64 *v6; // r12
  __int64 v8[2]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v9; // [rsp+10h] [rbp-40h]
  __int64 v10; // [rsp+18h] [rbp-38h]
  int v11; // [rsp+20h] [rbp-30h]
  __int64 v12; // [rsp+28h] [rbp-28h]
  __int64 v13; // [rsp+30h] [rbp-20h]
  __int64 v14; // [rsp+38h] [rbp-18h]

  v8[0] = a1;
  v13 = a4;
  v14 = a3;
  v8[1] = 0;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v6 = sub_14927C0(v8, a2, a5, a6);
  j___libc_free_0(v9);
  return v6;
}

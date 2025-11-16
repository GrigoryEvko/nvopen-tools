// Function: sub_13F2700
// Address: 0x13f2700
//
int *__fastcall sub_13F2700(int *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __m128i v9; // [rsp+0h] [rbp-40h] BYREF

  if ( *(_BYTE *)(a3 + 16) > 0x10u && !(unsigned __int8)sub_13E8A40(a2, a3, a4) )
  {
    v9.m128i_i64[0] = a4;
    v9.m128i_i64[1] = a3;
    sub_13ED650(a2, &v9);
    sub_13EFEC0(a2);
  }
  sub_13E9630(a1, a2, a3, a4);
  sub_13EE9C0(a2, a3, a1, a5);
  return a1;
}

// Function: sub_199E790
// Address: 0x199e790
//
char __fastcall sub_199E790(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        unsigned int a9,
        __int64 a10,
        unsigned __int8 a11)
{
  char result; // al
  unsigned __int64 v16; // r12
  __int64 v17; // [rsp+0h] [rbp-50h]
  unsigned __int8 v18; // [rsp+14h] [rbp-3Ch]

  v18 = a11;
  if ( sub_14560B0(a10) )
    return 1;
  v17 = sub_199D980((__int64)&a10, a2, a7, a8);
  v16 = sub_199E590((__int64)&a10, a2, a7, a8);
  result = sub_14560B0(a10);
  if ( result )
  {
    if ( v16 | v17 )
      return sub_1993620(a1, a3, a4, a5, a6, a9, v16, v17, v18, 2LL * (a5 != 3) - 1);
    return 1;
  }
  return result;
}

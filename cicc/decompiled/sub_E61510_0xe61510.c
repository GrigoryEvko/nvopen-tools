// Function: sub_E61510
// Address: 0xe61510
//
__int64 __fastcall sub_E61510(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        unsigned int a5,
        int a6,
        __int16 a7,
        char a8,
        char a9)
{
  __m128i v10; // [rsp+0h] [rbp-20h] BYREF
  int v11; // [rsp+10h] [rbp-10h]
  __int16 v12; // [rsp+14h] [rbp-Ch]
  char v13; // [rsp+16h] [rbp-Ah]

  v10.m128i_i64[0] = a3;
  v10.m128i_i64[1] = __PAIR64__(a5, a4);
  v12 = a7;
  v11 = a6;
  v13 = v13 & 0xFC | (a8 | (2 * a9)) & 3;
  return sub_E61330(a1, &v10);
}

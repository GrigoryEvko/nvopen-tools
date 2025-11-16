// Function: sub_1FD3790
// Address: 0x1fd3790
//
__int64 *__fastcall sub_1FD3790(__int64 *a1, __int32 a2, unsigned int a3, __int16 a4)
{
  __int64 v5; // rdi
  __int64 v6; // rsi
  __m128i v8; // [rsp+0h] [rbp-40h] BYREF
  __int64 v9; // [rsp+10h] [rbp-30h]
  __int64 v10; // [rsp+18h] [rbp-28h]
  __int64 v11; // [rsp+20h] [rbp-20h]

  v5 = a1[1];
  v8.m128i_i8[0] = 0;
  v8.m128i_i32[2] = a2;
  v6 = *a1;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v8.m128i_i8[3] = ((unsigned __int8)(a3 >> 9) << 7)
                 | (((a3 & 0x18) != 0) << 6)
                 | (32 * ((a3 & 4) != 0)) & 0x3F
                 | (16 * ((a3 & 2) != 0)) & 0x3F
                 | v8.m128i_i8[3] & 0xF;
  v8.m128i_i16[1] &= 0xF00Fu;
  v8.m128i_i8[4] = (8 * ((unsigned __int8)a3 >> 7))
                 | (4 * ((a3 & 0x40) != 0))
                 | (2 * (BYTE1(a3) & 1)) & 0xF3
                 | ((a3 & 0x20) != 0)
                 | v8.m128i_i8[4] & 0xF0;
  v8.m128i_i32[0] = ((a4 & 0xFFF) << 8) | v8.m128i_i32[0] & 0xFFF000FF;
  sub_1E1A9C0(v5, v6, &v8);
  return a1;
}

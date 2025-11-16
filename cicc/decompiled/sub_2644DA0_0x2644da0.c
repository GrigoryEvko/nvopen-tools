// Function: sub_2644DA0
// Address: 0x2644da0
//
__int64 *__fastcall sub_2644DA0(
        __int64 *a1,
        __int32 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        const __m128i a7)
{
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __m128i v11[2]; // [rsp+0h] [rbp-E0h] BYREF
  __int16 v12; // [rsp+20h] [rbp-C0h]
  __m128i v13[3]; // [rsp+30h] [rbp-B0h] BYREF
  __m128i v14[2]; // [rsp+60h] [rbp-80h] BYREF
  __int16 v15; // [rsp+80h] [rbp-60h]
  __m128i v16[5]; // [rsp+90h] [rbp-50h] BYREF

  if ( a2 )
  {
    v14[0].m128i_i32[0] = a2;
    v15 = 265;
    v12 = 260;
    v11[0].m128i_i64[0] = (__int64)&qword_4FF3200;
    sub_9C6370(v13, &a7, v11, a4, a5, a6);
    sub_9C6370(v16, v13, v14, v8, v9, v10);
    sub_CA0F50(a1, (void **)v16);
  }
  else
  {
    sub_CA0F50(a1, (void **)&a7);
  }
  return a1;
}

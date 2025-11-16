// Function: sub_370E590
// Address: 0x370e590
//
unsigned __int64 *__fastcall sub_370E590(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // rax
  unsigned __int16 v7; // [rsp+6h] [rbp-6Ah] BYREF
  unsigned __int64 v8; // [rsp+8h] [rbp-68h] BYREF
  __m128i v9[2]; // [rsp+10h] [rbp-60h] BYREF
  char v10; // [rsp+30h] [rbp-40h]
  char v11; // [rsp+31h] [rbp-3Fh]

  v7 = 0;
  v9[0].m128i_i64[0] = (__int64)"Padding";
  v11 = 1;
  v10 = 3;
  sub_370BC10(&v8, (_QWORD *)(a2 + 16), &v7, v9);
  v5 = v8 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v8 & 0xFFFFFFFFFFFFFFFELL) != 0
    || (v9[0].m128i_i64[0] = (__int64)"Type",
        v11 = 1,
        v10 = 3,
        sub_37011E0(&v8, (_QWORD *)(a2 + 16), (unsigned int *)(a4 + 2), v9[0].m128i_i64),
        v5 = v8 & 0xFFFFFFFFFFFFFFFELL,
        (v8 & 0xFFFFFFFFFFFFFFFELL) != 0) )
  {
    *a1 = v5 | 1;
  }
  else
  {
    *a1 = 1;
    v9[0].m128i_i64[0] = 0;
    sub_9C66B0(v9[0].m128i_i64);
  }
  return a1;
}

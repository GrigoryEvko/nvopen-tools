// Function: sub_3776C40
// Address: 0x3776c40
//
__int64 __fastcall sub_3776C40(__int64 a1, _QWORD *a2, __int128 *a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  __m128i v10; // xmm1
  __m128i v12; // [rsp+0h] [rbp-70h] BYREF
  __m128i v13; // [rsp+10h] [rbp-60h] BYREF
  __int64 v14[2]; // [rsp+20h] [rbp-50h] BYREF
  __m128i v15; // [rsp+30h] [rbp-40h] BYREF
  __m128i v16[3]; // [rsp+40h] [rbp-30h] BYREF

  v12.m128i_i16[0] = 0;
  v6 = *((unsigned int *)a3 + 2);
  v13.m128i_i16[0] = 0;
  v7 = *(_QWORD *)a3;
  v12.m128i_i64[1] = 0;
  v8 = *(_QWORD *)(v7 + 48) + 16 * v6;
  v13.m128i_i64[1] = 0;
  LOWORD(v7) = *(_WORD *)v8;
  v9 = *(_QWORD *)(v8 + 8);
  LOWORD(v14[0]) = v7;
  v14[1] = v9;
  sub_33D0340((__int64)&v15, (__int64)a2, v14);
  v10 = _mm_loadu_si128(v16);
  v12 = _mm_loadu_si128(&v15);
  v13 = v10;
  sub_3408290(a1, a2, a3, a4, (unsigned int *)&v12, (unsigned int *)&v13, v12);
  return a1;
}

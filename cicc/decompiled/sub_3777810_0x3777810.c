// Function: sub_3777810
// Address: 0x3777810
//
__m128i *__fastcall sub_3777810(
        __m128i *a1,
        __int64 *a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __m128i a6)
{
  __int64 v8; // rsi
  unsigned __int16 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r8
  _QWORD *v12; // r14
  __int64 v13; // rax
  __int16 v14; // dx
  __int64 v15; // rax
  __m128i v16; // xmm3
  __m128i v17; // xmm1
  unsigned __int128 v19; // [rsp+0h] [rbp-A0h] BYREF
  __m128i v20; // [rsp+10h] [rbp-90h] BYREF
  __m128i v21; // [rsp+20h] [rbp-80h] BYREF
  __m128i v22; // [rsp+30h] [rbp-70h] BYREF
  __m128i v23; // [rsp+40h] [rbp-60h] BYREF
  __int64 v24[2]; // [rsp+50h] [rbp-50h] BYREF
  __m128i v25; // [rsp+60h] [rbp-40h] BYREF
  __m128i v26; // [rsp+70h] [rbp-30h] BYREF

  v8 = *a2;
  v19 = __PAIR128__(a4, a3);
  v20.m128i_i32[2] = 0;
  v9 = (unsigned __int16 *)(*(_QWORD *)(a3 + 48) + 16LL * (unsigned int)a4);
  v10 = a2[1];
  v21.m128i_i32[2] = 0;
  v20.m128i_i64[0] = 0;
  v11 = *((_QWORD *)v9 + 1);
  v21.m128i_i64[0] = 0;
  sub_2FE6CC0((__int64)&v25, v8, *(_QWORD *)(v10 + 64), *v9, v11);
  if ( v25.m128i_i8[0] == 6 )
  {
    sub_375E8D0((__int64)a2, v19, *((__int64 *)&v19 + 1), (__int64)&v20, (__int64)&v21);
  }
  else
  {
    v22.m128i_i64[1] = 0;
    v12 = (_QWORD *)a2[1];
    v22.m128i_i16[0] = 0;
    v23.m128i_i16[0] = 0;
    v23.m128i_i64[1] = 0;
    v13 = *(_QWORD *)(v19 + 48) + 16LL * DWORD2(v19);
    v14 = *(_WORD *)v13;
    v15 = *(_QWORD *)(v13 + 8);
    LOWORD(v24[0]) = v14;
    v24[1] = v15;
    sub_33D0340((__int64)&v25, (__int64)v12, v24);
    v16 = _mm_loadu_si128(&v26);
    v22 = _mm_loadu_si128(&v25);
    v23 = v16;
    sub_3408290((__int64)&v25, v12, (__int128 *)&v19, a5, (unsigned int *)&v22, (unsigned int *)&v23, a6);
    v20.m128i_i64[0] = v25.m128i_i64[0];
    v20.m128i_i32[2] = v25.m128i_i32[2];
    v21.m128i_i64[0] = v26.m128i_i64[0];
    v21.m128i_i32[2] = v26.m128i_i32[2];
  }
  v17 = _mm_loadu_si128(&v21);
  *a1 = _mm_loadu_si128(&v20);
  a1[1] = v17;
  return a1;
}

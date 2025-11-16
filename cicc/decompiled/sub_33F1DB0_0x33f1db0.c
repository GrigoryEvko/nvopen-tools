// Function: sub_33F1DB0
// Address: 0x33f1db0
//
__m128i *__fastcall sub_33F1DB0(
        __int64 *a1,
        char a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        __int16 a6,
        __int128 a7,
        __int64 a8,
        __int64 a9,
        __int128 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13,
        __int16 a14,
        __int64 a15)
{
  unsigned __int16 *v16; // rax
  __int64 v17; // r8
  unsigned int v18; // ecx
  _QWORD *v19; // rax
  unsigned int v20; // edx
  __int64 v21; // r8
  __int64 v22; // r9
  _QWORD *v23; // r14
  __int64 v24; // r15
  unsigned __int8 v25; // al
  __int128 v27; // [rsp-60h] [rbp-F0h]
  unsigned int v28; // [rsp+8h] [rbp-88h]
  _QWORD *v29; // [rsp+10h] [rbp-80h]
  __int16 v33; // [rsp+38h] [rbp-58h]
  __m128i v34; // [rsp+40h] [rbp-50h] BYREF
  __int64 v35; // [rsp+50h] [rbp-40h]

  v33 = a14;
  v16 = (unsigned __int16 *)(*(_QWORD *)(a8 + 48) + 16LL * (unsigned int)a9);
  v17 = *((_QWORD *)v16 + 1);
  v18 = *v16;
  v34.m128i_i64[0] = 0;
  v34.m128i_i32[2] = 0;
  v19 = sub_33F17F0(a1, 51, (__int64)&v34, v18, v17);
  if ( v34.m128i_i64[0] )
  {
    v28 = v20;
    v29 = v19;
    sub_B91220((__int64)&v34, v34.m128i_i64[0]);
    v20 = v28;
    v19 = v29;
  }
  v23 = v19;
  v34 = _mm_loadu_si128((const __m128i *)&a10);
  v24 = v20;
  v35 = a11;
  v25 = sub_33CC4A0((__int64)a1, a12, a13, HIBYTE(a6), v21, v22);
  if ( HIBYTE(a6) )
    v25 = a6;
  *((_QWORD *)&v27 + 1) = v24;
  *(_QWORD *)&v27 = v23;
  return sub_33EA290(a1, 0, a2, a4, a5, a3, a7, a8, a9, v27, *(_OWORD *)&v34, v35, a12, a13, v25, v33, a15, 0);
}

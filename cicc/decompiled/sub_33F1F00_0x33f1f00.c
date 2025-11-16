// Function: sub_33F1F00
// Address: 0x33f1f00
//
__m128i *__fastcall sub_33F1F00(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int128 a9,
        __int64 a10,
        __int16 a11,
        __int16 a12,
        __int64 a13,
        __int64 a14)
{
  unsigned __int16 *v15; // rax
  __int64 v16; // r8
  unsigned int v17; // ecx
  _QWORD *v18; // rax
  unsigned int v19; // edx
  __int64 v20; // r9
  _QWORD *v21; // r14
  __int64 v22; // r15
  unsigned __int8 v23; // al
  __int128 v25; // [rsp-60h] [rbp-100h]
  unsigned int v26; // [rsp+8h] [rbp-98h]
  char v27; // [rsp+8h] [rbp-98h]
  _QWORD *v28; // [rsp+10h] [rbp-90h]
  __int128 v29; // [rsp+20h] [rbp-80h]
  __int16 v31; // [rsp+40h] [rbp-60h]
  __m128i v32; // [rsp+50h] [rbp-50h] BYREF
  __int64 v33; // [rsp+60h] [rbp-40h]

  *(_QWORD *)&v29 = a5;
  v31 = a12;
  v15 = (unsigned __int16 *)(*(_QWORD *)(a7 + 48) + 16LL * (unsigned int)a8);
  v16 = *((_QWORD *)v15 + 1);
  v17 = *v15;
  *((_QWORD *)&v29 + 1) = a6;
  v32.m128i_i64[0] = 0;
  v32.m128i_i32[2] = 0;
  v18 = sub_33F17F0(a1, 51, (__int64)&v32, v17, v16);
  if ( v32.m128i_i64[0] )
  {
    v26 = v19;
    v28 = v18;
    sub_B91220((__int64)&v32, v32.m128i_i64[0]);
    v19 = v26;
    v18 = v28;
  }
  v21 = v18;
  v32 = _mm_loadu_si128((const __m128i *)&a9);
  v22 = v19;
  v33 = a10;
  v27 = HIBYTE(a11);
  v23 = sub_33CC4A0((__int64)a1, a2, a3, HIBYTE(a11), a2, v20);
  if ( v27 )
    v23 = a11;
  *((_QWORD *)&v25 + 1) = v22;
  *(_QWORD *)&v25 = v21;
  return sub_33EA290(a1, 0, 0, a2, a3, a4, v29, a7, a8, v25, *(_OWORD *)&v32, v33, a2, a3, v23, v31, a13, a14);
}

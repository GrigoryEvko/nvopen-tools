// Function: sub_33E6260
// Address: 0x33e6260
//
_QWORD *__fastcall sub_33E6260(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int32 a4,
        const void *a5,
        __int64 a6,
        char a7,
        __int64 *a8,
        int a9)
{
  __int64 *v11; // r12
  __int64 v12; // rsi
  __int64 *v13; // r8
  _QWORD *v14; // rax
  _QWORD *v15; // r14
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rbx
  unsigned __int64 v20; // rax
  __int64 v21; // rsi
  __m128i v22; // xmm0
  __int64 v23; // rdx
  __m128i *v24; // rax
  __int64 v28; // [rsp+18h] [rbp-58h] BYREF
  __m128i v29; // [rsp+20h] [rbp-50h] BYREF
  __int64 v30; // [rsp+30h] [rbp-40h]

  v11 = *(__int64 **)(a1 + 720);
  v12 = *a8;
  v29.m128i_i32[0] = 2;
  v13 = v11;
  v29.m128i_i32[2] = a4;
  v28 = v12;
  if ( v12 )
  {
    sub_B96E90((__int64)&v28, v12, 1);
    v13 = *(__int64 **)(a1 + 720);
  }
  v14 = (_QWORD *)sub_A777F0(0x40u, v13);
  v15 = v14;
  if ( v14 )
  {
    *v14 = 1;
    v16 = *v11;
    v11[10] += 24;
    v17 = (v16 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v11[1] >= v17 + 24 && v16 )
      *v11 = v17 + 24;
    else
      v17 = sub_9D1E70((__int64)v11, 24, 24, 3);
    v15[1] = v17;
    v15[2] = a6;
    v18 = *v11;
    v19 = 8 * a6;
    v11[10] += v19;
    v20 = (v18 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v11[1] >= v19 + v20 && v18 )
      *v11 = v19 + v20;
    else
      v20 = sub_9D1E70((__int64)v11, v19, v19, 3);
    v15[3] = v20;
    v21 = v28;
    v15[4] = a2;
    v15[6] = v21;
    v15[5] = a3;
    if ( v21 )
      sub_B96E90((__int64)(v15 + 6), v21, 1);
    v22 = _mm_loadu_si128(&v29);
    *((_BYTE *)v15 + 60) = a7;
    v23 = v30;
    *((_BYTE *)v15 + 61) = 0;
    *((_DWORD *)v15 + 14) = a9;
    *((_WORD *)v15 + 31) = 0;
    v24 = (__m128i *)v15[1];
    v24[1].m128i_i64[0] = v23;
    *v24 = v22;
    if ( v19 )
      memmove((void *)v15[3], a5, v19);
  }
  if ( v28 )
    sub_B91220((__int64)&v28, v28);
  return v15;
}

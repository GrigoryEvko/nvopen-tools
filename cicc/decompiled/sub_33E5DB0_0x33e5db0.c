// Function: sub_33E5DB0
// Address: 0x33e5db0
//
_QWORD *__fastcall sub_33E5DB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, int a6)
{
  __int64 v9; // rsi
  __int64 *v10; // r13
  __int64 *v11; // r8
  _QWORD *v12; // rax
  _QWORD *v13; // r12
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rsi
  __m128i *v18; // rax
  __int64 v19; // rdx
  __m128i v20; // xmm0
  __int64 v22; // [rsp+8h] [rbp-58h] BYREF
  __m128i v23; // [rsp+10h] [rbp-50h] BYREF
  __int64 v24; // [rsp+20h] [rbp-40h]

  v9 = *a5;
  v10 = *(__int64 **)(a1 + 720);
  v23.m128i_i64[1] = a4;
  v23.m128i_i32[0] = 1;
  v22 = v9;
  v11 = v10;
  if ( v9 )
  {
    sub_B96E90((__int64)&v22, v9, 1);
    v11 = *(__int64 **)(a1 + 720);
  }
  v12 = (_QWORD *)sub_A777F0(0x40u, v11);
  v13 = v12;
  if ( v12 )
  {
    *v12 = 1;
    v14 = *v10;
    v10[10] += 24;
    v15 = (v14 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v10[1] >= v15 + 24 && v14 )
      *v10 = v15 + 24;
    else
      v15 = sub_9D1E70((__int64)v10, 24, 24, 3);
    v13[1] = v15;
    v13[2] = 0;
    v16 = (*v10 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v10[1] >= v16 && *v10 )
      *v10 = v16;
    else
      v16 = sub_9D1E70((__int64)v10, 0, 0, 3);
    v17 = v22;
    v13[3] = v16;
    v13[4] = a2;
    v13[5] = a3;
    v13[6] = v17;
    if ( v17 )
      sub_B96E90((__int64)(v13 + 6), v17, 1);
    v18 = (__m128i *)v13[1];
    v19 = v24;
    *((_DWORD *)v13 + 14) = a6;
    *((_DWORD *)v13 + 15) = 0;
    v20 = _mm_loadu_si128(&v23);
    v18[1].m128i_i64[0] = v19;
    *v18 = v20;
  }
  if ( v22 )
    sub_B91220((__int64)&v22, v22);
  return v13;
}

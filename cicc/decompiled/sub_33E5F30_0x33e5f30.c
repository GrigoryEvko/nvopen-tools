// Function: sub_33E5F30
// Address: 0x33e5f30
//
_QWORD *__fastcall sub_33E5F30(__int64 a1, __int64 a2, __int64 a3, __int32 a4, char a5, __int64 *a6, int a7)
{
  __int64 *v10; // r13
  __int64 v11; // rsi
  __int64 *v12; // r8
  _QWORD *v13; // rax
  _QWORD *v14; // r12
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rdx
  __m128i v20; // xmm0
  __m128i *v21; // rax
  __int64 v23; // [rsp+8h] [rbp-58h] BYREF
  __m128i v24; // [rsp+10h] [rbp-50h] BYREF
  __int64 v25; // [rsp+20h] [rbp-40h]

  v10 = *(__int64 **)(a1 + 720);
  v11 = *a6;
  v24.m128i_i32[2] = a4;
  v24.m128i_i32[0] = 3;
  v23 = v11;
  v12 = v10;
  if ( v11 )
  {
    sub_B96E90((__int64)&v23, v11, 1);
    v12 = *(__int64 **)(a1 + 720);
  }
  v13 = (_QWORD *)sub_A777F0(0x40u, v12);
  v14 = v13;
  if ( v13 )
  {
    *v13 = 1;
    v15 = *v10;
    v10[10] += 24;
    v16 = (v15 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v10[1] >= v16 + 24 && v15 )
      *v10 = v16 + 24;
    else
      v16 = sub_9D1E70((__int64)v10, 24, 24, 3);
    v14[1] = v16;
    v14[2] = 0;
    v17 = (*v10 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v10[1] >= v17 && *v10 )
      *v10 = v17;
    else
      v17 = sub_9D1E70((__int64)v10, 0, 0, 3);
    v18 = v23;
    v14[3] = v17;
    v14[4] = a2;
    v14[5] = a3;
    v14[6] = v18;
    if ( v18 )
      sub_B96E90((__int64)(v14 + 6), v18, 1);
    v19 = v25;
    *((_BYTE *)v14 + 60) = a5;
    *((_BYTE *)v14 + 61) = 0;
    v20 = _mm_loadu_si128(&v24);
    *((_DWORD *)v14 + 14) = a7;
    *((_WORD *)v14 + 31) = 0;
    v21 = (__m128i *)v14[1];
    v21[1].m128i_i64[0] = v19;
    *v21 = v20;
  }
  if ( v23 )
    sub_B91220((__int64)&v23, v23);
  return v14;
}

// Function: sub_381DC70
// Address: 0x381dc70
//
void __fastcall sub_381DC70(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rsi
  __int64 v9; // rax
  _QWORD *v10; // rdi
  unsigned int *v11; // rcx
  const __m128i *v12; // rax
  __m128i v13; // xmm1
  bool v14; // zf
  __int64 v15; // rdx
  unsigned __int8 *v16; // rax
  __m128i v17; // xmm3
  __m128i v18; // xmm4
  int v19; // edx
  _QWORD *v20; // rdi
  __int64 v21; // rsi
  unsigned __int8 *v22; // rax
  int v23; // edx
  __int128 v24; // [rsp-20h] [rbp-110h]
  __int128 v25; // [rsp-10h] [rbp-100h]
  unsigned int *v26; // [rsp+10h] [rbp-E0h]
  __int64 v27; // [rsp+18h] [rbp-D8h]
  __m128i v28; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v29; // [rsp+50h] [rbp-A0h] BYREF
  __m128i v30; // [rsp+60h] [rbp-90h] BYREF
  __m128i v31; // [rsp+70h] [rbp-80h] BYREF
  __int64 v32; // [rsp+80h] [rbp-70h] BYREF
  int v33; // [rsp+88h] [rbp-68h]
  __m128i v34; // [rsp+90h] [rbp-60h] BYREF
  __m128i v35; // [rsp+A0h] [rbp-50h]
  __m128i v36; // [rsp+B0h] [rbp-40h]

  v8 = *(_QWORD *)(a2 + 80);
  v28.m128i_i64[0] = 0;
  v28.m128i_i32[2] = 0;
  v29.m128i_i64[0] = 0;
  v29.m128i_i32[2] = 0;
  v30.m128i_i64[0] = 0;
  v30.m128i_i32[2] = 0;
  v31.m128i_i64[0] = 0;
  v31.m128i_i32[2] = 0;
  v32 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v32, v8, 1);
  v33 = *(_DWORD *)(a2 + 72);
  sub_375E510(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), (__int64)&v28, (__int64)&v29);
  sub_375E510(
    a1,
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
    (__int64)&v30,
    (__int64)&v31);
  v9 = sub_33E5110(
         *(__int64 **)(a1 + 8),
         *(unsigned __int16 *)(*(_QWORD *)(v28.m128i_i64[0] + 48) + 16LL * v28.m128i_u32[2]),
         *(_QWORD *)(*(_QWORD *)(v28.m128i_i64[0] + 48) + 16LL * v28.m128i_u32[2] + 8),
         *(unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL),
         *(_QWORD *)(*(_QWORD *)(a2 + 48) + 24LL));
  v10 = *(_QWORD **)(a1 + 8);
  v11 = (unsigned int *)v9;
  v12 = *(const __m128i **)(a2 + 40);
  v13 = _mm_loadu_si128(&v30);
  v14 = *(_DWORD *)(a2 + 24) == 74;
  v34 = _mm_loadu_si128(&v28);
  v35 = v13;
  *((_QWORD *)&v25 + 1) = 3;
  *(_QWORD *)&v25 = &v34;
  v36 = _mm_loadu_si128(v12 + 5);
  v26 = v11;
  v27 = v15;
  v16 = sub_3411630(v10, (unsigned int)!v14 + 72, (__int64)&v32, v11, v15, (__int64)&v34, v25);
  v17 = _mm_loadu_si128(&v29);
  v36.m128i_i32[2] = 1;
  v18 = _mm_loadu_si128(&v31);
  *(_QWORD *)a3 = v16;
  v34 = v17;
  *(_DWORD *)(a3 + 8) = v19;
  v20 = *(_QWORD **)(a1 + 8);
  *((_QWORD *)&v24 + 1) = 3;
  v21 = *(unsigned int *)(a2 + 24);
  *(_QWORD *)&v24 = &v34;
  v35 = v18;
  v36.m128i_i64[0] = (__int64)v16;
  v22 = sub_3411630(v20, v21, (__int64)&v32, v26, v27, (__int64)&v34, v24);
  *(_QWORD *)a4 = v22;
  *(_DWORD *)(a4 + 8) = v23;
  sub_3760E70(a1, a2, 1, (unsigned __int64)v22, 1);
  if ( v32 )
    sub_B91220((__int64)&v32, v32);
}

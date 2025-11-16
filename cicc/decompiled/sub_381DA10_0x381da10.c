// Function: sub_381DA10
// Address: 0x381da10
//
void __fastcall sub_381DA10(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rsi
  __int64 v9; // rax
  __m128i v10; // xmm0
  __m128i v11; // xmm1
  unsigned int *v12; // rcx
  const __m128i *v13; // rax
  __m128i v14; // xmm3
  _QWORD *v15; // rdi
  __m128i v16; // xmm2
  __m128i v17; // xmm4
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // r9
  unsigned __int8 *v21; // rax
  int v22; // edx
  _QWORD *v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // r9
  unsigned __int8 *v26; // rax
  int v27; // edx
  __int128 v28; // [rsp-20h] [rbp-130h]
  __int128 v29; // [rsp-10h] [rbp-120h]
  unsigned int *v30; // [rsp+0h] [rbp-110h]
  __int64 v31; // [rsp+8h] [rbp-108h]
  __m128i v32; // [rsp+30h] [rbp-E0h] BYREF
  __m128i v33; // [rsp+40h] [rbp-D0h] BYREF
  __m128i v34; // [rsp+50h] [rbp-C0h] BYREF
  __m128i v35; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v36; // [rsp+70h] [rbp-A0h] BYREF
  int v37; // [rsp+78h] [rbp-98h]
  _OWORD v38[3]; // [rsp+80h] [rbp-90h] BYREF
  _OWORD v39[2]; // [rsp+B0h] [rbp-60h] BYREF
  unsigned __int8 *v40; // [rsp+D0h] [rbp-40h]
  int v41; // [rsp+D8h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 80);
  v32.m128i_i64[0] = 0;
  v32.m128i_i32[2] = 0;
  v33.m128i_i64[0] = 0;
  v33.m128i_i32[2] = 0;
  v34.m128i_i64[0] = 0;
  v34.m128i_i32[2] = 0;
  v35.m128i_i64[0] = 0;
  v35.m128i_i32[2] = 0;
  v36 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v36, v8, 1);
  v37 = *(_DWORD *)(a2 + 72);
  sub_375E510(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), (__int64)&v32, (__int64)&v33);
  sub_375E510(
    a1,
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
    (__int64)&v34,
    (__int64)&v35);
  v9 = sub_33E5110(
         *(__int64 **)(a1 + 8),
         *(unsigned __int16 *)(*(_QWORD *)(v32.m128i_i64[0] + 48) + 16LL * v32.m128i_u32[2]),
         *(_QWORD *)(*(_QWORD *)(v32.m128i_i64[0] + 48) + 16LL * v32.m128i_u32[2] + 8),
         *(unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL),
         *(_QWORD *)(*(_QWORD *)(a2 + 48) + 24LL));
  v10 = _mm_loadu_si128(&v32);
  v11 = _mm_loadu_si128(&v34);
  v40 = 0;
  v12 = (unsigned int *)v9;
  v13 = *(const __m128i **)(a2 + 40);
  v14 = _mm_loadu_si128(&v33);
  v15 = *(_QWORD **)(a1 + 8);
  v38[1] = v11;
  v16 = _mm_loadu_si128(v13 + 5);
  *((_QWORD *)&v29 + 1) = 3;
  v17 = _mm_loadu_si128(&v35);
  v18 = *(unsigned int *)(a2 + 24);
  *(_QWORD *)&v29 = v38;
  v38[0] = v10;
  v38[2] = v16;
  v39[0] = v14;
  v39[1] = v17;
  v30 = v12;
  v31 = v19;
  v41 = 0;
  v21 = sub_3411630(v15, v18, (__int64)&v36, v12, v19, v20, v29);
  v41 = 1;
  *(_QWORD *)a3 = v21;
  *(_DWORD *)(a3 + 8) = v22;
  v23 = *(_QWORD **)(a1 + 8);
  v24 = *(unsigned int *)(a2 + 24);
  v40 = v21;
  *((_QWORD *)&v28 + 1) = 3;
  *(_QWORD *)&v28 = v39;
  v26 = sub_3411630(v23, v24, (__int64)&v36, v30, v31, v25, v28);
  *(_QWORD *)a4 = v26;
  *(_DWORD *)(a4 + 8) = v27;
  sub_3760E70(a1, a2, 1, (unsigned __int64)v26, 1);
  if ( v36 )
    sub_B91220((__int64)&v36, v36);
}

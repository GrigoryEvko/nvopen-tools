// Function: sub_381D0D0
// Address: 0x381d0d0
//
void __fastcall sub_381D0D0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rsi
  unsigned __int16 *v9; // rax
  __int64 v10; // rax
  __m128i v11; // xmm0
  __m128i v12; // xmm1
  unsigned int *v13; // rcx
  const __m128i *v14; // rax
  __m128i v15; // xmm3
  _QWORD *v16; // rdi
  __m128i v17; // xmm2
  __m128i v18; // xmm4
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // r9
  unsigned __int8 *v22; // rax
  int v23; // edx
  _QWORD *v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // r9
  unsigned __int8 *v27; // rax
  int v28; // edx
  __int128 v29; // [rsp-20h] [rbp-130h]
  __int128 v30; // [rsp-10h] [rbp-120h]
  unsigned int *v31; // [rsp+0h] [rbp-110h]
  __int64 v32; // [rsp+8h] [rbp-108h]
  __m128i v33; // [rsp+30h] [rbp-E0h] BYREF
  __m128i v34; // [rsp+40h] [rbp-D0h] BYREF
  __m128i v35; // [rsp+50h] [rbp-C0h] BYREF
  __m128i v36; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v37; // [rsp+70h] [rbp-A0h] BYREF
  int v38; // [rsp+78h] [rbp-98h]
  _OWORD v39[3]; // [rsp+80h] [rbp-90h] BYREF
  _OWORD v40[2]; // [rsp+B0h] [rbp-60h] BYREF
  unsigned __int8 *v41; // [rsp+D0h] [rbp-40h]
  int v42; // [rsp+D8h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 80);
  v33.m128i_i64[0] = 0;
  v33.m128i_i32[2] = 0;
  v34.m128i_i64[0] = 0;
  v34.m128i_i32[2] = 0;
  v35.m128i_i64[0] = 0;
  v35.m128i_i32[2] = 0;
  v36.m128i_i64[0] = 0;
  v36.m128i_i32[2] = 0;
  v37 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v37, v8, 1);
  v38 = *(_DWORD *)(a2 + 72);
  sub_375E510(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), (__int64)&v33, (__int64)&v34);
  sub_375E510(
    a1,
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
    (__int64)&v35,
    (__int64)&v36);
  v9 = (unsigned __int16 *)(*(_QWORD *)(v33.m128i_i64[0] + 48) + 16LL * v33.m128i_u32[2]);
  v10 = sub_33E5110(*(__int64 **)(a1 + 8), *v9, *((_QWORD *)v9 + 1), 262, 0);
  v11 = _mm_loadu_si128(&v33);
  v12 = _mm_loadu_si128(&v35);
  v41 = 0;
  v13 = (unsigned int *)v10;
  v14 = *(const __m128i **)(a2 + 40);
  v15 = _mm_loadu_si128(&v34);
  v16 = *(_QWORD **)(a1 + 8);
  v39[1] = v12;
  v17 = _mm_loadu_si128(v14 + 5);
  *((_QWORD *)&v30 + 1) = 3;
  v18 = _mm_loadu_si128(&v36);
  v19 = *(unsigned int *)(a2 + 24);
  *(_QWORD *)&v30 = v39;
  v39[0] = v11;
  v39[2] = v17;
  v40[0] = v15;
  v40[1] = v18;
  v31 = v13;
  v32 = v20;
  v42 = 0;
  v22 = sub_3411630(v16, v19, (__int64)&v37, v13, v20, v21, v30);
  v42 = 1;
  *(_QWORD *)a3 = v22;
  *(_DWORD *)(a3 + 8) = v23;
  v24 = *(_QWORD **)(a1 + 8);
  v25 = *(unsigned int *)(a2 + 24);
  v41 = v22;
  *((_QWORD *)&v29 + 1) = 3;
  *(_QWORD *)&v29 = v40;
  v27 = sub_3411630(v24, v25, (__int64)&v37, v31, v32, v26, v29);
  *(_QWORD *)a4 = v27;
  *(_DWORD *)(a4 + 8) = v28;
  sub_3760E70(a1, a2, 1, (unsigned __int64)v27, 1);
  if ( v37 )
    sub_B91220((__int64)&v37, v37);
}

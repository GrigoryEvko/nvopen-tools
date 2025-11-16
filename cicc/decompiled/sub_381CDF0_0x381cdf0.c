// Function: sub_381CDF0
// Address: 0x381cdf0
//
void __fastcall sub_381CDF0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rsi
  unsigned __int16 *v9; // rax
  unsigned int *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r9
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  bool v15; // zf
  _QWORD *v16; // rdi
  __m128i v17; // xmm2
  __m128i v18; // xmm3
  unsigned __int8 *v19; // rax
  int v20; // edx
  _QWORD *v21; // rdi
  __int64 v22; // r9
  int v23; // edx
  unsigned __int8 *v24; // rax
  int v25; // edx
  _QWORD *v26; // rdi
  __int64 v27; // r9
  __int128 v28; // [rsp-20h] [rbp-140h]
  __int128 v29; // [rsp-20h] [rbp-140h]
  __int128 v30; // [rsp-10h] [rbp-130h]
  unsigned int *v31; // [rsp+0h] [rbp-120h]
  __int64 v32; // [rsp+8h] [rbp-118h]
  __m128i v33; // [rsp+50h] [rbp-D0h] BYREF
  __m128i v34; // [rsp+60h] [rbp-C0h] BYREF
  __m128i v35; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v36; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v37; // [rsp+90h] [rbp-90h] BYREF
  int v38; // [rsp+98h] [rbp-88h]
  _OWORD v39[2]; // [rsp+A0h] [rbp-80h] BYREF
  _OWORD v40[2]; // [rsp+C0h] [rbp-60h] BYREF
  unsigned __int8 *v41; // [rsp+E0h] [rbp-40h]
  int v42; // [rsp+E8h] [rbp-38h]

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
  v10 = (unsigned int *)sub_33E5110(*(__int64 **)(a1 + 8), *v9, *((_QWORD *)v9 + 1), 262, 0);
  v13 = _mm_loadu_si128(&v33);
  v14 = _mm_loadu_si128(&v35);
  v41 = 0;
  v15 = *(_DWORD *)(a2 + 24) == 68;
  v16 = *(_QWORD **)(a1 + 8);
  v17 = _mm_loadu_si128(&v34);
  v18 = _mm_loadu_si128(&v36);
  v39[0] = v13;
  v42 = 0;
  v39[1] = v14;
  v40[0] = v17;
  v40[1] = v18;
  *((_QWORD *)&v30 + 1) = 2;
  *(_QWORD *)&v30 = v39;
  v31 = v10;
  v32 = v11;
  if ( v15 )
  {
    v24 = sub_3411630(v16, 68, (__int64)&v37, v10, v11, v12, v30);
    *(_QWORD *)a3 = v24;
    *(_DWORD *)(a3 + 8) = v25;
    v26 = *(_QWORD **)(a1 + 8);
    v41 = v24;
    *((_QWORD *)&v29 + 1) = 3;
    *(_QWORD *)&v29 = v40;
    v42 = 1;
    *(_QWORD *)a4 = sub_3411630(v26, 70, (__int64)&v37, v31, v32, v27, v29);
  }
  else
  {
    v19 = sub_3411630(v16, 69, (__int64)&v37, v10, v11, v12, v30);
    *(_QWORD *)a3 = v19;
    *(_DWORD *)(a3 + 8) = v20;
    v21 = *(_QWORD **)(a1 + 8);
    v41 = v19;
    *((_QWORD *)&v28 + 1) = 3;
    *(_QWORD *)&v28 = v40;
    v42 = 1;
    *(_QWORD *)a4 = sub_3411630(v21, 71, (__int64)&v37, v31, v32, v22, v28);
  }
  *(_DWORD *)(a4 + 8) = v23;
  sub_3760E70(a1, a2, 1, *(_QWORD *)a4, 1);
  if ( v37 )
    sub_B91220((__int64)&v37, v37);
}

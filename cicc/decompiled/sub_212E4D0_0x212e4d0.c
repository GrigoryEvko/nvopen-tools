// Function: sub_212E4D0
// Address: 0x212e4d0
//
void __fastcall sub_212E4D0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rsi
  unsigned __int8 *v9; // rax
  __int64 v10; // rax
  __m128i v11; // xmm0
  __m128i v12; // xmm1
  const void ***v13; // rcx
  const __m128i *v14; // rax
  __m128i v15; // xmm3
  __int64 *v16; // rdi
  __m128i v17; // xmm2
  __m128i v18; // xmm4
  __int64 v19; // rsi
  int v20; // edx
  __int64 v21; // r9
  __int64 *v22; // rax
  int v23; // edx
  __int64 *v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // r9
  __int64 *v27; // rax
  int v28; // edx
  const __m128i *v29; // r9
  __int128 v30; // [rsp-20h] [rbp-130h]
  __int128 v31; // [rsp-10h] [rbp-120h]
  const void ***v32; // [rsp+0h] [rbp-110h]
  int v33; // [rsp+8h] [rbp-108h]
  __m128i v34; // [rsp+30h] [rbp-E0h] BYREF
  __m128i v35; // [rsp+40h] [rbp-D0h] BYREF
  __m128i v36; // [rsp+50h] [rbp-C0h] BYREF
  __m128i v37; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v38; // [rsp+70h] [rbp-A0h] BYREF
  int v39; // [rsp+78h] [rbp-98h]
  _OWORD v40[3]; // [rsp+80h] [rbp-90h] BYREF
  _OWORD v41[2]; // [rsp+B0h] [rbp-60h] BYREF
  __int64 *v42; // [rsp+D0h] [rbp-40h]
  int v43; // [rsp+D8h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 72);
  v34.m128i_i64[0] = 0;
  v34.m128i_i32[2] = 0;
  v35.m128i_i64[0] = 0;
  v35.m128i_i32[2] = 0;
  v36.m128i_i64[0] = 0;
  v36.m128i_i32[2] = 0;
  v37.m128i_i64[0] = 0;
  v37.m128i_i32[2] = 0;
  v38 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v38, v8, 2);
  v39 = *(_DWORD *)(a2 + 64);
  sub_20174B0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), &v34, &v35);
  sub_20174B0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL), &v36, &v37);
  v9 = (unsigned __int8 *)(*(_QWORD *)(v34.m128i_i64[0] + 40) + 16LL * v34.m128i_u32[2]);
  v10 = sub_1D252B0(*(_QWORD *)(a1 + 8), *v9, *((_QWORD *)v9 + 1), 111, 0);
  v11 = _mm_loadu_si128(&v34);
  v12 = _mm_loadu_si128(&v36);
  v42 = 0;
  v13 = (const void ***)v10;
  v14 = *(const __m128i **)(a2 + 32);
  v15 = _mm_loadu_si128(&v35);
  v16 = *(__int64 **)(a1 + 8);
  v40[0] = v11;
  v17 = _mm_loadu_si128(v14 + 5);
  *((_QWORD *)&v31 + 1) = 3;
  v18 = _mm_loadu_si128(&v37);
  v19 = *(unsigned __int16 *)(a2 + 24);
  *(_QWORD *)&v31 = v40;
  v40[1] = v12;
  v40[2] = v17;
  v41[0] = v15;
  v41[1] = v18;
  v32 = v13;
  v33 = v20;
  v43 = 0;
  v22 = sub_1D36D80(
          v16,
          v19,
          (__int64)&v38,
          v13,
          v20,
          *(double *)v11.m128i_i64,
          *(double *)v12.m128i_i64,
          v17,
          v21,
          v31);
  v43 = 1;
  *(_QWORD *)a3 = v22;
  *(_DWORD *)(a3 + 8) = v23;
  v24 = *(__int64 **)(a1 + 8);
  v25 = *(unsigned __int16 *)(a2 + 24);
  v42 = v22;
  *((_QWORD *)&v30 + 1) = 3;
  *(_QWORD *)&v30 = v41;
  v27 = sub_1D36D80(
          v24,
          v25,
          (__int64)&v38,
          v32,
          v33,
          *(double *)v11.m128i_i64,
          *(double *)v12.m128i_i64,
          v17,
          v26,
          v30);
  *(_QWORD *)a4 = v27;
  *(_DWORD *)(a4 + 8) = v28;
  sub_2013400(a1, a2, 1, (__int64)v27, (__m128i *)1, v29);
  if ( v38 )
    sub_161E7C0((__int64)&v38, v38);
}

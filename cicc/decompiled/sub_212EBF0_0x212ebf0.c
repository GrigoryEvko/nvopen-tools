// Function: sub_212EBF0
// Address: 0x212ebf0
//
void __fastcall sub_212EBF0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rsi
  __int64 v9; // rax
  __m128i v10; // xmm0
  __m128i v11; // xmm1
  const void ***v12; // rcx
  const __m128i *v13; // rax
  __m128i v14; // xmm3
  __int64 *v15; // rdi
  __m128i v16; // xmm2
  __m128i v17; // xmm4
  __int64 v18; // rsi
  int v19; // edx
  __int64 v20; // r9
  __int64 *v21; // rax
  int v22; // edx
  __int64 *v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // r9
  __int64 *v26; // rax
  int v27; // edx
  const __m128i *v28; // r9
  __int128 v29; // [rsp-20h] [rbp-130h]
  __int128 v30; // [rsp-10h] [rbp-120h]
  const void ***v31; // [rsp+0h] [rbp-110h]
  int v32; // [rsp+8h] [rbp-108h]
  __m128i v33; // [rsp+30h] [rbp-E0h] BYREF
  __m128i v34; // [rsp+40h] [rbp-D0h] BYREF
  __m128i v35; // [rsp+50h] [rbp-C0h] BYREF
  __m128i v36; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v37; // [rsp+70h] [rbp-A0h] BYREF
  int v38; // [rsp+78h] [rbp-98h]
  _OWORD v39[3]; // [rsp+80h] [rbp-90h] BYREF
  _OWORD v40[2]; // [rsp+B0h] [rbp-60h] BYREF
  __int64 *v41; // [rsp+D0h] [rbp-40h]
  int v42; // [rsp+D8h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 72);
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
    sub_1623A60((__int64)&v37, v8, 2);
  v38 = *(_DWORD *)(a2 + 64);
  sub_20174B0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), &v33, &v34);
  sub_20174B0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL), &v35, &v36);
  v9 = sub_1D252B0(
         *(_QWORD *)(a1 + 8),
         *(unsigned __int8 *)(*(_QWORD *)(v33.m128i_i64[0] + 40) + 16LL * v33.m128i_u32[2]),
         *(_QWORD *)(*(_QWORD *)(v33.m128i_i64[0] + 40) + 16LL * v33.m128i_u32[2] + 8),
         *(unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL),
         *(_QWORD *)(*(_QWORD *)(a2 + 40) + 24LL));
  v10 = _mm_loadu_si128(&v33);
  v11 = _mm_loadu_si128(&v35);
  v41 = 0;
  v12 = (const void ***)v9;
  v13 = *(const __m128i **)(a2 + 32);
  v14 = _mm_loadu_si128(&v34);
  v15 = *(__int64 **)(a1 + 8);
  v39[0] = v10;
  v16 = _mm_loadu_si128(v13 + 5);
  *((_QWORD *)&v30 + 1) = 3;
  v17 = _mm_loadu_si128(&v36);
  v18 = *(unsigned __int16 *)(a2 + 24);
  *(_QWORD *)&v30 = v39;
  v39[1] = v11;
  v39[2] = v16;
  v40[0] = v14;
  v40[1] = v17;
  v31 = v12;
  v32 = v19;
  v42 = 0;
  v21 = sub_1D36D80(
          v15,
          v18,
          (__int64)&v37,
          v12,
          v19,
          *(double *)v10.m128i_i64,
          *(double *)v11.m128i_i64,
          v16,
          v20,
          v30);
  v42 = 1;
  *(_QWORD *)a3 = v21;
  *(_DWORD *)(a3 + 8) = v22;
  v23 = *(__int64 **)(a1 + 8);
  v24 = *(unsigned __int16 *)(a2 + 24);
  v41 = v21;
  *((_QWORD *)&v29 + 1) = 3;
  *(_QWORD *)&v29 = v40;
  v26 = sub_1D36D80(
          v23,
          v24,
          (__int64)&v37,
          v31,
          v32,
          *(double *)v10.m128i_i64,
          *(double *)v11.m128i_i64,
          v16,
          v25,
          v29);
  *(_QWORD *)a4 = v26;
  *(_DWORD *)(a4 + 8) = v27;
  sub_2013400(a1, a2, 1, (__int64)v26, (__m128i *)1, v28);
  if ( v37 )
    sub_161E7C0((__int64)&v37, v37);
}

// Function: sub_212E1F0
// Address: 0x212e1f0
//
void __fastcall sub_212E1F0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rsi
  unsigned __int8 *v9; // rax
  const void ***v10; // rax
  int v11; // edx
  __int64 v12; // r9
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  bool v16; // zf
  __m128i v17; // xmm3
  __int64 *v18; // rdi
  __int64 *v19; // rax
  int v20; // edx
  __int64 *v21; // rdi
  __int64 v22; // r9
  int v23; // edx
  const __m128i *v24; // r9
  __int64 *v25; // rax
  int v26; // edx
  __int64 *v27; // rdi
  __int64 v28; // r9
  __int128 v29; // [rsp-20h] [rbp-140h]
  __int128 v30; // [rsp-20h] [rbp-140h]
  __int128 v31; // [rsp-10h] [rbp-130h]
  const void ***v32; // [rsp+0h] [rbp-120h]
  int v33; // [rsp+8h] [rbp-118h]
  __m128i v34; // [rsp+50h] [rbp-D0h] BYREF
  __m128i v35; // [rsp+60h] [rbp-C0h] BYREF
  __m128i v36; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v37; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v38; // [rsp+90h] [rbp-90h] BYREF
  int v39; // [rsp+98h] [rbp-88h]
  _OWORD v40[2]; // [rsp+A0h] [rbp-80h] BYREF
  _OWORD v41[2]; // [rsp+C0h] [rbp-60h] BYREF
  __int64 *v42; // [rsp+E0h] [rbp-40h]
  int v43; // [rsp+E8h] [rbp-38h]

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
  v10 = (const void ***)sub_1D252B0(*(_QWORD *)(a1 + 8), *v9, *((_QWORD *)v9 + 1), 111, 0);
  v13 = _mm_loadu_si128(&v34);
  v14 = _mm_loadu_si128(&v36);
  v42 = 0;
  v15 = _mm_loadu_si128(&v35);
  v16 = *(_WORD *)(a2 + 24) == 64;
  v17 = _mm_loadu_si128(&v37);
  v43 = 0;
  v18 = *(__int64 **)(a1 + 8);
  v40[0] = v13;
  v40[1] = v14;
  v41[0] = v15;
  v41[1] = v17;
  *((_QWORD *)&v31 + 1) = 2;
  *(_QWORD *)&v31 = v40;
  v32 = v10;
  v33 = v11;
  if ( v16 )
  {
    v25 = sub_1D36D80(
            v18,
            64,
            (__int64)&v38,
            v10,
            v11,
            *(double *)v13.m128i_i64,
            *(double *)v14.m128i_i64,
            v15,
            v12,
            v31);
    *(_QWORD *)a3 = v25;
    *(_DWORD *)(a3 + 8) = v26;
    v27 = *(__int64 **)(a1 + 8);
    v42 = v25;
    *((_QWORD *)&v30 + 1) = 3;
    *(_QWORD *)&v30 = v41;
    v43 = 1;
    *(_QWORD *)a4 = sub_1D36D80(
                      v27,
                      66,
                      (__int64)&v38,
                      v32,
                      v33,
                      *(double *)v13.m128i_i64,
                      *(double *)v14.m128i_i64,
                      v15,
                      v28,
                      v30);
  }
  else
  {
    v19 = sub_1D36D80(
            v18,
            65,
            (__int64)&v38,
            v10,
            v11,
            *(double *)v13.m128i_i64,
            *(double *)v14.m128i_i64,
            v15,
            v12,
            v31);
    *(_QWORD *)a3 = v19;
    *(_DWORD *)(a3 + 8) = v20;
    v21 = *(__int64 **)(a1 + 8);
    v42 = v19;
    *((_QWORD *)&v29 + 1) = 3;
    *(_QWORD *)&v29 = v41;
    v43 = 1;
    *(_QWORD *)a4 = sub_1D36D80(
                      v21,
                      67,
                      (__int64)&v38,
                      v32,
                      v33,
                      *(double *)v13.m128i_i64,
                      *(double *)v14.m128i_i64,
                      v15,
                      v22,
                      v29);
  }
  *(_DWORD *)(a4 + 8) = v23;
  sub_2013400(a1, a2, 1, *(_QWORD *)a4, (__m128i *)1, v24);
  if ( v38 )
    sub_161E7C0((__int64)&v38, v38);
}

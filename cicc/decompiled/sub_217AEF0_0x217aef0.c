// Function: sub_217AEF0
// Address: 0x217aef0
//
__int64 *__fastcall sub_217AEF0(const __m128i *a1, __m128i a2, double a3, __m128i a4)
{
  __int64 v4; // r13
  const __m128i *v5; // r12
  __int64 *v6; // r15
  __m128i v7; // rax
  const void ***v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 *v14; // r14
  __int64 *v15; // rax
  __int64 v16; // r9
  __int64 *v17; // r12
  __int64 v18; // rsi
  __int128 v19; // rax
  __int64 v20; // r9
  _QWORD *v21; // rax
  __m128i v22; // xmm0
  __m128i v23; // xmm1
  __int64 v24; // rdx
  __int64 v25; // r9
  __int128 v27; // [rsp-28h] [rbp-110h]
  __int128 v28; // [rsp-8h] [rbp-F0h]
  int v29; // [rsp+8h] [rbp-E0h]
  const void ***v30; // [rsp+10h] [rbp-D8h]
  __m128i v31; // [rsp+18h] [rbp-D0h] BYREF
  const void ***v32; // [rsp+28h] [rbp-C0h]
  __int64 v33; // [rsp+30h] [rbp-B8h]
  const __m128i *v34; // [rsp+38h] [rbp-B0h]
  __int64 *v35; // [rsp+40h] [rbp-A8h]
  __int64 *v36; // [rsp+48h] [rbp-A0h]
  unsigned __int64 v37; // [rsp+50h] [rbp-98h]
  __int64 *v38; // [rsp+58h] [rbp-90h]
  unsigned __int64 v39; // [rsp+60h] [rbp-88h]
  __int64 *v40; // [rsp+68h] [rbp-80h] BYREF
  int v41; // [rsp+70h] [rbp-78h]
  _QWORD *v42; // [rsp+78h] [rbp-70h]
  __int64 v43; // [rsp+80h] [rbp-68h]
  __m128i v44; // [rsp+88h] [rbp-60h]
  __m128i v45; // [rsp+98h] [rbp-50h]
  __int64 *v46; // [rsp+A8h] [rbp-40h]
  int v47; // [rsp+B0h] [rbp-38h]

  v4 = (__int64)&a1[3].m128i_i64[1];
  v5 = a1;
  v6 = (__int64 *)a1->m128i_i64[0];
  v7.m128i_i64[0] = sub_1D38BB0(
                      a1->m128i_i64[0],
                      a1->m128i_u32[2],
                      (__int64)&a1[3].m128i_i64[1],
                      5,
                      0,
                      0,
                      a2,
                      a3,
                      a4,
                      0);
  v31 = v7;
  v8 = (const void ***)sub_1D252B0((__int64)v6, 1, 0, 111, 0);
  v29 = v9;
  *((_QWORD *)&v27 + 1) = 1;
  *(_QWORD *)&v27 = a1[1].m128i_i64[0];
  v33 = v9;
  v30 = v8;
  v14 = sub_1D37440(
          v6,
          283,
          (__int64)&a1[3].m128i_i64[1],
          v8,
          v9,
          v10,
          *(double *)a2.m128i_i64,
          a3,
          a4,
          v27,
          *(_OWORD *)&v31);
  v15 = (__int64 *)a1[2].m128i_i64[0];
  v35 = v15;
  v16 = (__int64)(v15 + 1);
  if ( (__int64 *)a1[2].m128i_i64[1] != v15 + 1 )
  {
    v34 = a1;
    v17 = v15 + 1;
    v32 = v30;
    do
    {
      v18 = *(v17 - 1);
      v35 = v17;
      v36 = v14;
      ++v17;
      v37 = v37 & 0xFFFFFFFF00000000LL | 1;
      *(_QWORD *)&v19 = sub_1D2A490(v6, v18, v11, v12, v13, v16);
      v38 = v14;
      v39 &= 0xFFFFFFFF00000000LL;
      v14 = sub_1D37470(
              v6,
              284,
              v4,
              v32,
              v33,
              v20,
              __PAIR128__(v39, (unsigned __int64)v14),
              v19,
              __PAIR128__(v37, (unsigned __int64)v14));
    }
    while ( (__int64 *)v34[2].m128i_i64[1] != v17 );
    v5 = v34;
  }
  v40 = v14;
  v41 = 0;
  v21 = sub_1D2A490(v6, *v35, v11, v12, v13, v16);
  v22 = _mm_loadu_si128(v5 + 1);
  v23 = _mm_load_si128(&v31);
  v42 = v21;
  v43 = v24;
  *((_QWORD *)&v28 + 1) = 5;
  *(_QWORD *)&v28 = &v40;
  v46 = v14;
  v47 = 1;
  v44 = v22;
  v45 = v23;
  return sub_1D36D80(v6, 285, v4, v30, v29, *(double *)v22.m128i_i64, *(double *)v23.m128i_i64, a4, v25, v28);
}

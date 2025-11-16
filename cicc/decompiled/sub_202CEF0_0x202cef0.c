// Function: sub_202CEF0
// Address: 0x202cef0
//
__int64 *__fastcall sub_202CEF0(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rdx
  char v6; // r13
  const __m128i *v7; // rax
  __int64 v8; // rcx
  __m128i v9; // xmm0
  __int64 v10; // rcx
  __int64 v11; // rdx
  __m128i v12; // xmm1
  unsigned __int16 v13; // r14
  __int64 v14; // rdx
  __int64 v15; // rsi
  __m128i v16; // xmm2
  __m128i v17; // xmm3
  int v18; // r15d
  char v19; // al
  _QWORD *v20; // rdi
  unsigned int v21; // r15d
  int v22; // r13d
  __int64 v23; // rax
  int v24; // edx
  __int64 v25; // r13
  __int128 v26; // rax
  __int64 *v27; // rax
  _QWORD *v28; // r13
  unsigned int v29; // edx
  int v30; // edx
  __int64 *v31; // r12
  __int64 *v33; // [rsp+18h] [rbp-188h]
  char v34; // [rsp+2Fh] [rbp-171h]
  char v35; // [rsp+3Ch] [rbp-164h]
  unsigned int v36; // [rsp+3Ch] [rbp-164h]
  __int64 v37; // [rsp+40h] [rbp-160h]
  __int64 v38; // [rsp+48h] [rbp-158h]
  __int64 v39; // [rsp+50h] [rbp-150h]
  unsigned __int64 v40; // [rsp+58h] [rbp-148h]
  __int64 v41; // [rsp+B0h] [rbp-F0h] BYREF
  int v42; // [rsp+B8h] [rbp-E8h]
  _QWORD v43[2]; // [rsp+C0h] [rbp-E0h] BYREF
  __int64 v44; // [rsp+D0h] [rbp-D0h] BYREF
  unsigned __int64 v45; // [rsp+D8h] [rbp-C8h]
  __int128 v46; // [rsp+E0h] [rbp-C0h] BYREF
  __m128i v47; // [rsp+F0h] [rbp-B0h] BYREF
  __m128i v48; // [rsp+100h] [rbp-A0h] BYREF
  __m128i v49; // [rsp+110h] [rbp-90h] BYREF
  __int64 v50; // [rsp+120h] [rbp-80h]
  __int128 v51; // [rsp+130h] [rbp-70h] BYREF
  __int64 v52; // [rsp+140h] [rbp-60h]
  __m128i v53; // [rsp+150h] [rbp-50h] BYREF
  __m128i v54[4]; // [rsp+160h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a2 + 72);
  v41 = v4;
  if ( v4 )
    sub_1623A60((__int64)&v41, v4, 2);
  v5 = *(_QWORD *)(a2 + 96);
  v6 = *(_BYTE *)(a2 + 27);
  v42 = *(_DWORD *)(a2 + 64);
  v7 = *(const __m128i **)(a2 + 32);
  v8 = v7->m128i_i64[0];
  v9 = _mm_loadu_si128(v7 + 5);
  v44 = 0;
  *(_QWORD *)&v46 = 0;
  v38 = v8;
  v10 = v7->m128i_i64[1];
  v43[1] = v5;
  v11 = *(_QWORD *)(a2 + 104);
  v37 = v10;
  LOBYTE(v10) = *(_BYTE *)(a2 + 88);
  v12 = _mm_loadu_si128((const __m128i *)(v11 + 40));
  v13 = *(_WORD *)(v11 + 32);
  LODWORD(v45) = 0;
  LOBYTE(v43[0]) = v10;
  LOWORD(v10) = *(_WORD *)(v11 + 34);
  v14 = *(_QWORD *)(v11 + 56);
  v49 = v12;
  DWORD2(v46) = 0;
  v50 = v14;
  v35 = v10;
  sub_2017DE0(a1, v7[2].m128i_u64[1], v7[3].m128i_i64[0], &v44, &v46);
  v15 = *(_QWORD *)(a1 + 8);
  v47.m128i_i8[0] = 0;
  v47.m128i_i64[1] = 0;
  sub_1D19A30((__int64)&v53, v15, v43);
  v16 = _mm_loadu_si128(&v53);
  v17 = _mm_loadu_si128(v54);
  v47 = v16;
  v48 = v17;
  if ( v53.m128i_i8[0] )
    v18 = sub_2021900(v53.m128i_i8[0]);
  else
    v18 = sub_1F58D40((__int64)&v47);
  if ( (v18 & 7) != 0
    || (v48.m128i_i8[0] ? (v19 = sub_2021900(v48.m128i_i8[0])) : (v19 = sub_1F58D40((__int64)&v48)), (v19 & 7) != 0) )
  {
    v31 = (__int64 *)sub_20B91E0(*(_QWORD *)a1, a2, *(_QWORD *)(a1 + 8));
  }
  else
  {
    v20 = *(_QWORD **)(a1 + 8);
    v21 = (unsigned int)(v18 + 7) >> 3;
    v22 = v6 & 4;
    v36 = (unsigned int)(1 << v35) >> 1;
    v23 = *(_QWORD *)(a2 + 104);
    v34 = v22;
    if ( v22 )
      v44 = sub_1D2C750(
              v20,
              v38,
              v37,
              (__int64)&v41,
              v44,
              v45,
              v9.m128i_i64[0],
              v9.m128i_i64[1],
              *(_OWORD *)v23,
              *(_QWORD *)(v23 + 16),
              v47.m128i_i64[0],
              v47.m128i_i64[1],
              v36,
              v13,
              (__int64)&v49);
    else
      v44 = sub_1D2BF40(
              v20,
              v38,
              v37,
              (__int64)&v41,
              v44,
              v45,
              v9.m128i_i64[0],
              v9.m128i_i64[1],
              *(_OWORD *)v23,
              *(_QWORD *)(v23 + 16),
              v36,
              v13,
              (__int64)&v49);
    LODWORD(v45) = v24;
    v25 = 16LL * v9.m128i_u32[2];
    v33 = *(__int64 **)(a1 + 8);
    *(_QWORD *)&v26 = sub_1D38BB0(
                        (__int64)v33,
                        v21,
                        (__int64)&v41,
                        *(unsigned __int8 *)(v25 + *(_QWORD *)(v9.m128i_i64[0] + 40)),
                        *(const void ***)(v25 + *(_QWORD *)(v9.m128i_i64[0] + 40) + 8),
                        0,
                        v9,
                        *(double *)v12.m128i_i64,
                        v16,
                        0);
    v27 = sub_1D332F0(
            v33,
            52,
            (__int64)&v41,
            *(unsigned __int8 *)(*(_QWORD *)(v9.m128i_i64[0] + 40) + v25),
            *(const void ***)(*(_QWORD *)(v9.m128i_i64[0] + 40) + v25 + 8),
            3u,
            *(double *)v9.m128i_i64,
            *(double *)v12.m128i_i64,
            v16,
            v9.m128i_i64[0],
            v9.m128i_u32[2],
            v26);
    v28 = *(_QWORD **)(a1 + 8);
    v39 = (__int64)v27;
    v40 = v29 | v9.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    if ( v34 )
    {
      sub_1F7DDA0((__int64)&v51, *(_QWORD *)(a2 + 104), v21);
      *(_QWORD *)&v46 = sub_1D2C750(
                          v28,
                          v38,
                          v37,
                          (__int64)&v41,
                          v46,
                          *((__int64 *)&v46 + 1),
                          v39,
                          v40,
                          v51,
                          v52,
                          v48.m128i_i64[0],
                          v48.m128i_i64[1],
                          v36,
                          v13,
                          (__int64)&v49);
    }
    else
    {
      sub_1F7DDA0((__int64)&v53, *(_QWORD *)(a2 + 104), v21);
      *(_QWORD *)&v46 = sub_1D2BF40(
                          v28,
                          v38,
                          v37,
                          (__int64)&v41,
                          v46,
                          *((__int64 *)&v46 + 1),
                          v39,
                          v40,
                          *(_OWORD *)&v53,
                          v54[0].m128i_i64[0],
                          v36,
                          v13,
                          (__int64)&v49);
    }
    DWORD2(v46) = v30;
    v31 = sub_1D332F0(
            *(__int64 **)(a1 + 8),
            2,
            (__int64)&v41,
            1,
            0,
            0,
            *(double *)v9.m128i_i64,
            *(double *)v12.m128i_i64,
            v16,
            v44,
            v45,
            v46);
  }
  if ( v41 )
    sub_161E7C0((__int64)&v41, v41);
  return v31;
}

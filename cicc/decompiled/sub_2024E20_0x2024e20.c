// Function: sub_2024E20
// Address: 0x2024e20
//
unsigned __int64 __fastcall sub_2024E20(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        double a5,
        __m128i a6,
        __m128i a7)
{
  unsigned __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rsi
  int v13; // eax
  __int64 v14; // rsi
  __int64 v15; // rax
  __m128i v16; // xmm0
  unsigned __int8 *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r11
  __int64 v20; // rsi
  __int64 v21; // rdx
  int v22; // eax
  __int64 v23; // rax
  char v24; // dl
  __int64 v25; // rax
  const void ***v26; // rax
  __int64 *v27; // rax
  int v28; // edx
  __int64 v29; // rdx
  const void ***v30; // rax
  __int64 *v31; // rax
  __int64 v32; // rsi
  unsigned int v33; // edx
  unsigned __int64 result; // rax
  __int64 v35; // [rsp+8h] [rbp-128h]
  __int64 v36; // [rsp+10h] [rbp-120h]
  __int64 *v37; // [rsp+10h] [rbp-120h]
  int v38; // [rsp+38h] [rbp-F8h]
  __int64 v39; // [rsp+40h] [rbp-F0h] BYREF
  unsigned __int64 v40; // [rsp+48h] [rbp-E8h]
  __int64 v41; // [rsp+50h] [rbp-E0h] BYREF
  unsigned __int64 v42; // [rsp+58h] [rbp-D8h]
  __int64 v43; // [rsp+60h] [rbp-D0h] BYREF
  int v44; // [rsp+68h] [rbp-C8h]
  __int128 v45; // [rsp+70h] [rbp-C0h] BYREF
  __int128 v46; // [rsp+80h] [rbp-B0h] BYREF
  __m128i v47; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v48; // [rsp+A0h] [rbp-90h] BYREF
  int v49; // [rsp+A8h] [rbp-88h]
  __m128 v50; // [rsp+B0h] [rbp-80h] BYREF
  __m128i v51; // [rsp+C0h] [rbp-70h] BYREF
  _QWORD v52[2]; // [rsp+D0h] [rbp-60h] BYREF
  __m128i v53; // [rsp+E0h] [rbp-50h] BYREF
  __m128i v54; // [rsp+F0h] [rbp-40h] BYREF

  v10 = *(unsigned __int64 **)(a2 + 32);
  LODWORD(v40) = 0;
  LODWORD(v42) = 0;
  v39 = 0;
  v11 = v10[1];
  v41 = 0;
  sub_2017DE0(a1, *v10, v11, &v39, &v41);
  v12 = *(_QWORD *)(a2 + 72);
  v43 = v12;
  if ( v12 )
    sub_1623A60((__int64)&v43, v12, 2);
  v13 = *(_DWORD *)(a2 + 64);
  v14 = *(_QWORD *)a1;
  DWORD2(v45) = 0;
  DWORD2(v46) = 0;
  v44 = v13;
  v15 = *(_QWORD *)(a2 + 32);
  *(_QWORD *)&v45 = 0;
  *(_QWORD *)&v46 = 0;
  v16 = _mm_loadu_si128((const __m128i *)(v15 + 40));
  v47 = v16;
  v17 = (unsigned __int8 *)(*(_QWORD *)(v16.m128i_i64[0] + 40) + 16LL * v16.m128i_u32[2]);
  sub_1F40D10((__int64)&v53, v14, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v17, *((_QWORD *)v17 + 1));
  if ( v53.m128i_i8[0] == 6 )
  {
    sub_2017DE0(a1, v47.m128i_u64[0], v47.m128i_i64[1], &v45, &v46);
  }
  else
  {
    v18 = v47.m128i_i64[0];
    v19 = *(_QWORD *)(a1 + 8);
    v20 = *(_QWORD *)(v47.m128i_i64[0] + 72);
    v48 = v20;
    if ( v20 )
    {
      v35 = v47.m128i_i64[0];
      v36 = v19;
      sub_1623A60((__int64)&v48, v20, 2);
      v21 = v47.m128i_i64[0];
      v19 = v36;
      v18 = v35;
    }
    else
    {
      v21 = v47.m128i_i64[0];
    }
    v22 = *(_DWORD *)(v18 + 64);
    v50.m128_i8[0] = 0;
    v50.m128_u64[1] = 0;
    v49 = v22;
    v51.m128i_i64[1] = 0;
    v23 = *(_QWORD *)(v21 + 40) + 16LL * v47.m128i_u32[2];
    v51.m128i_i8[0] = 0;
    v24 = *(_BYTE *)v23;
    v25 = *(_QWORD *)(v23 + 8);
    v37 = (__int64 *)v19;
    LOBYTE(v52[0]) = v24;
    v52[1] = v25;
    sub_1D19A30((__int64)&v53, v19, v52);
    a6 = _mm_loadu_si128(&v53);
    a7 = _mm_loadu_si128(&v54);
    v50 = (__m128)a6;
    v51 = a7;
    sub_1D40600(
      (__int64)&v53,
      v37,
      (__int64)&v47,
      (__int64)&v48,
      (const void ***)&v50,
      (const void ***)&v51,
      v16,
      *(double *)a6.m128i_i64,
      a7);
    *(_QWORD *)&v45 = v53.m128i_i64[0];
    DWORD2(v45) = v53.m128i_i32[2];
    *(_QWORD *)&v46 = v54.m128i_i64[0];
    DWORD2(v46) = v54.m128i_i32[2];
    if ( v48 )
      sub_161E7C0((__int64)&v48, v48);
  }
  v26 = (const void ***)(*(_QWORD *)(v39 + 40) + 16LL * (unsigned int)v40);
  v27 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          101,
          (__int64)&v43,
          *(unsigned __int8 *)v26,
          v26[1],
          0,
          *(double *)v16.m128i_i64,
          *(double *)a6.m128i_i64,
          a7,
          v39,
          v40,
          v45);
  v38 = v28;
  v29 = v41;
  *(_QWORD *)a3 = v27;
  *(_DWORD *)(a3 + 8) = v38;
  v30 = (const void ***)(*(_QWORD *)(v29 + 40) + 16LL * (unsigned int)v42);
  v31 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          101,
          (__int64)&v43,
          *(unsigned __int8 *)v30,
          v30[1],
          0,
          *(double *)v16.m128i_i64,
          *(double *)a6.m128i_i64,
          a7,
          v41,
          v42,
          v46);
  v32 = v43;
  *(_QWORD *)a4 = v31;
  result = v33;
  *(_DWORD *)(a4 + 8) = v33;
  if ( v32 )
    return sub_161E7C0((__int64)&v43, v32);
  return result;
}

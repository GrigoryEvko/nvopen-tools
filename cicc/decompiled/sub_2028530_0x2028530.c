// Function: sub_2028530
// Address: 0x2028530
//
unsigned __int64 __fastcall sub_2028530(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128i a5,
        __m128i a6,
        __m128i a7)
{
  __int64 v8; // rsi
  __int64 v9; // rsi
  char *v10; // rax
  char v11; // dl
  __int64 v12; // rsi
  unsigned int *v13; // rax
  unsigned __int8 *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // r11
  unsigned int *v17; // r10
  int v18; // eax
  __int64 v19; // rdx
  char v20; // cl
  unsigned __int8 *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // r11
  __int64 v24; // rcx
  int v25; // eax
  __int64 v26; // rax
  char v27; // dl
  __int64 v28; // rax
  __m128i v29; // xmm3
  int v30; // edx
  __int64 *v31; // rax
  __int64 v32; // rsi
  unsigned int v33; // edx
  unsigned __int64 result; // rax
  __int64 v35; // [rsp+8h] [rbp-148h]
  __int64 v36; // [rsp+10h] [rbp-140h]
  __int64 *v37; // [rsp+10h] [rbp-140h]
  __int64 v38; // [rsp+10h] [rbp-140h]
  __int64 v39; // [rsp+10h] [rbp-140h]
  const void **v40; // [rsp+18h] [rbp-138h]
  unsigned __int64 v41; // [rsp+20h] [rbp-130h]
  const void **v42; // [rsp+28h] [rbp-128h]
  unsigned __int64 v43; // [rsp+30h] [rbp-120h]
  __int64 *v46; // [rsp+48h] [rbp-108h]
  __int64 v47; // [rsp+70h] [rbp-E0h] BYREF
  int v48; // [rsp+78h] [rbp-D8h]
  unsigned __int64 v49; // [rsp+80h] [rbp-D0h] BYREF
  __int16 *v50; // [rsp+88h] [rbp-C8h]
  unsigned __int64 v51; // [rsp+90h] [rbp-C0h] BYREF
  __int16 *v52; // [rsp+98h] [rbp-B8h]
  __int128 v53; // [rsp+A0h] [rbp-B0h] BYREF
  __int128 v54; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v55; // [rsp+C0h] [rbp-90h] BYREF
  int v56; // [rsp+C8h] [rbp-88h]
  __m128i v57; // [rsp+D0h] [rbp-80h] BYREF
  __m128 v58; // [rsp+E0h] [rbp-70h] BYREF
  __int64 v59; // [rsp+F0h] [rbp-60h] BYREF
  __int64 v60; // [rsp+F8h] [rbp-58h]
  __m128i v61; // [rsp+100h] [rbp-50h] BYREF
  __m128i v62; // [rsp+110h] [rbp-40h] BYREF

  v8 = *(_QWORD *)(a2 + 72);
  v47 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v47, v8, 2);
  v9 = *(_QWORD *)(a1 + 8);
  v48 = *(_DWORD *)(a2 + 64);
  v10 = *(char **)(a2 + 40);
  v11 = *v10;
  v60 = *((_QWORD *)v10 + 1);
  LOBYTE(v59) = v11;
  sub_1D19A30((__int64)&v61, v9, &v59);
  v12 = *(_QWORD *)a1;
  LODWORD(v50) = 0;
  v43 = v61.m128i_i64[0];
  LODWORD(v52) = 0;
  v42 = (const void **)v61.m128i_i64[1];
  DWORD2(v53) = 0;
  v41 = v62.m128i_i64[0];
  DWORD2(v54) = 0;
  v40 = (const void **)v62.m128i_i64[1];
  v13 = *(unsigned int **)(a2 + 32);
  v49 = 0;
  v51 = 0;
  *(_QWORD *)&v53 = 0;
  *(_QWORD *)&v54 = 0;
  v14 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v13 + 40LL) + 16LL * v13[2]);
  sub_1F40D10((__int64)&v61, v12, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v14, *((_QWORD *)v14 + 1));
  if ( v61.m128i_i8[0] == 6 )
  {
    sub_2017DE0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), &v49, &v51);
  }
  else
  {
    v15 = *(_QWORD *)(a2 + 72);
    v16 = *(_QWORD *)(a1 + 8);
    v55 = v15;
    if ( v15 )
    {
      v36 = v16;
      sub_1623A60((__int64)&v55, v15, 2);
      v16 = v36;
    }
    v17 = *(unsigned int **)(a2 + 32);
    v18 = *(_DWORD *)(a2 + 64);
    v57.m128i_i8[0] = 0;
    v57.m128i_i64[1] = 0;
    v56 = v18;
    v58.m128_i8[0] = 0;
    v58.m128_u64[1] = 0;
    v35 = (__int64)v17;
    v37 = (__int64 *)v16;
    v19 = *(_QWORD *)(*(_QWORD *)v17 + 40LL) + 16LL * v17[2];
    v20 = *(_BYTE *)v19;
    v60 = *(_QWORD *)(v19 + 8);
    LOBYTE(v59) = v20;
    sub_1D19A30((__int64)&v61, v16, &v59);
    a5 = _mm_loadu_si128(&v61);
    a6 = _mm_loadu_si128(&v62);
    v57 = a5;
    v58 = (__m128)a6;
    sub_1D40600(
      (__int64)&v61,
      v37,
      v35,
      (__int64)&v55,
      (const void ***)&v57,
      (const void ***)&v58,
      a5,
      *(double *)a6.m128i_i64,
      a7);
    if ( v55 )
      sub_161E7C0((__int64)&v55, v55);
    v49 = v61.m128i_i64[0];
    LODWORD(v50) = v61.m128i_i32[2];
    v51 = v62.m128i_i64[0];
    LODWORD(v52) = v62.m128i_i32[2];
  }
  v21 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 40LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL));
  sub_1F40D10((__int64)&v61, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v21, *((_QWORD *)v21 + 1));
  if ( v61.m128i_i8[0] == 6 )
  {
    sub_2017DE0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL), &v53, &v54);
  }
  else
  {
    v22 = *(_QWORD *)(a2 + 72);
    v23 = *(_QWORD *)(a1 + 8);
    v55 = v22;
    if ( v22 )
    {
      v38 = v23;
      sub_1623A60((__int64)&v55, v22, 2);
      v23 = v38;
    }
    v24 = *(_QWORD *)(a2 + 32);
    v25 = *(_DWORD *)(a2 + 64);
    v57.m128i_i8[0] = 0;
    v57.m128i_i64[1] = 0;
    v56 = v25;
    v58.m128_i8[0] = 0;
    v58.m128_u64[1] = 0;
    v39 = v24;
    v26 = *(_QWORD *)(*(_QWORD *)(v24 + 40) + 40LL) + 16LL * *(unsigned int *)(v24 + 48);
    v27 = *(_BYTE *)v26;
    v28 = *(_QWORD *)(v26 + 8);
    LOBYTE(v59) = v27;
    v60 = v28;
    v46 = (__int64 *)v23;
    sub_1D19A30((__int64)&v61, v23, &v59);
    a7 = _mm_loadu_si128(&v61);
    v29 = _mm_loadu_si128(&v62);
    v57 = a7;
    v58 = (__m128)v29;
    sub_1D40600(
      (__int64)&v61,
      v46,
      v39 + 40,
      (__int64)&v55,
      (const void ***)&v57,
      (const void ***)&v58,
      a5,
      *(double *)a6.m128i_i64,
      a7);
    if ( v55 )
      sub_161E7C0((__int64)&v55, v55);
    *(_QWORD *)&v53 = v61.m128i_i64[0];
    DWORD2(v53) = v61.m128i_i32[2];
    *(_QWORD *)&v54 = v62.m128i_i64[0];
    DWORD2(v54) = v62.m128i_i32[2];
  }
  *(_QWORD *)a3 = sub_1D3A900(
                    *(__int64 **)(a1 + 8),
                    *(unsigned __int16 *)(a2 + 24),
                    (__int64)&v47,
                    v43,
                    v42,
                    0,
                    (__m128)a5,
                    *(double *)a6.m128i_i64,
                    a7,
                    v49,
                    v50,
                    v53,
                    *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL),
                    *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  *(_DWORD *)(a3 + 8) = v30;
  v31 = sub_1D3A900(
          *(__int64 **)(a1 + 8),
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v47,
          v41,
          v40,
          0,
          (__m128)a5,
          *(double *)a6.m128i_i64,
          a7,
          v51,
          v52,
          v54,
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v32 = v47;
  *(_QWORD *)a4 = v31;
  result = v33;
  *(_DWORD *)(a4 + 8) = v33;
  if ( v32 )
    return sub_161E7C0((__int64)&v47, v32);
  return result;
}

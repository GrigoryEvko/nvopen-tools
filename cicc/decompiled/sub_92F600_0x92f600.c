// Function: sub_92F600
// Address: 0x92f600
//
__int64 __fastcall sub_92F600(__int64 *a1, __int64 a2, __int64 a3)
{
  _DWORD *v5; // r15
  __int64 v6; // r10
  _DWORD *v7; // r14
  __int64 v8; // r13
  __int64 v9; // rsi
  __m128i v10; // xmm0
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  __int64 v13; // r8
  __int64 v14; // r9
  char v15; // al
  __int64 v16; // rax
  __int64 (__fastcall *v17)(__int64 *, __int64, __m128i *, __int64, __int64); // r9
  __int64 v18; // r8
  __int64 v19; // rcx
  char v20; // al
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rdi
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 result; // rax
  __int64 v31; // rsi
  __m128i v32; // xmm4
  __m128i v33; // xmm5
  __m128i v34; // xmm3
  __int64 *v35; // [rsp+0h] [rbp-160h]
  __int64 *v36; // [rsp+8h] [rbp-158h]
  char v37; // [rsp+14h] [rbp-14Ch]
  __int64 v39; // [rsp+20h] [rbp-140h]
  __m128i *v40; // [rsp+28h] [rbp-138h]
  __int64 v41; // [rsp+30h] [rbp-130h]
  __int32 v42; // [rsp+30h] [rbp-130h]
  unsigned __int64 v43; // [rsp+38h] [rbp-128h]
  __int64 v44; // [rsp+48h] [rbp-118h] BYREF
  __int64 v45[5]; // [rsp+50h] [rbp-110h] BYREF
  int v46; // [rsp+78h] [rbp-E8h]
  char v47; // [rsp+7Ch] [rbp-E4h]
  __int64 v48; // [rsp+80h] [rbp-E0h]
  __int64 v49; // [rsp+90h] [rbp-D0h] BYREF
  int v50; // [rsp+98h] [rbp-C8h]
  char v51; // [rsp+9Ch] [rbp-C4h]
  __int64 v52; // [rsp+A0h] [rbp-C0h]
  __m128i v53; // [rsp+B0h] [rbp-B0h] BYREF
  __m128i v54; // [rsp+C0h] [rbp-A0h] BYREF
  __m128i v55; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v56; // [rsp+E0h] [rbp-80h]
  __m128i v57; // [rsp+F0h] [rbp-70h]
  __m128i v58; // [rsp+100h] [rbp-60h]
  __m128i v59; // [rsp+110h] [rbp-50h]
  __int64 v60; // [rsp+120h] [rbp-40h]

  v5 = (_DWORD *)(a2 + 36);
  v6 = *(_QWORD *)(a2 + 72);
  v7 = (_DWORD *)(v6 + 36);
  v41 = v6;
  v35 = *(__int64 **)(v6 + 16);
  v40 = sub_92CBF0(a1, (__int64)v35, a3);
  v43 = *(_QWORD *)a2;
  v8 = sub_91B7A0(a2);
  v36 = (__int64 *)v41;
  sub_926800((__int64)&v53, *a1, v41);
  v9 = *a1;
  v10 = _mm_loadu_si128(&v53);
  v11 = _mm_loadu_si128(&v54);
  v12 = _mm_loadu_si128(&v55);
  v60 = v56;
  v57 = v10;
  v58 = v11;
  v59 = v12;
  v42 = v53.m128i_i32[0];
  v37 = v56;
  sub_9286A0(
    (__int64)&v49,
    v9,
    v7,
    (unsigned int)v56,
    v13,
    v14,
    v10.m128i_i64[0],
    v10.m128i_u64[1],
    v11.m128i_u64[0],
    v11.m128i_i64[1],
    v12.m128i_i64[0],
    v12.m128i_i64[1],
    v56);
  v39 = v49;
  v15 = sub_91B6F0(v43);
  v16 = sub_92C930(a1, v39, v15, v8, v5);
  v17 = (__int64 (__fastcall *)(__int64 *, __int64, __m128i *, __int64, __int64))a3;
  if ( (unsigned __int8)(*(_BYTE *)(a2 + 56) - 84) <= 1u )
  {
    if ( (a3 & 1) != 0 )
      v17 = *(__int64 (__fastcall **)(__int64 *, __int64, __m128i *, __int64, __int64))(*a1 + a3 - 1);
    v18 = *v36;
    v19 = *v35;
  }
  else
  {
    if ( (a3 & 1) != 0 )
      v17 = *(__int64 (__fastcall **)(__int64 *, __int64, __m128i *, __int64, __int64))(*a1 + a3 - 1);
    v18 = 0;
    v19 = v8;
  }
  v44 = v17(a1, v16, v40, v19, v18);
  v20 = sub_91B6F0(v8);
  v21 = sub_92C930(a1, v44, v20, v43, v5);
  v26 = *a1;
  v44 = v21;
  if ( v42 == 1 )
  {
    if ( (v37 & 1) == 0 )
    {
      v57.m128i_i8[12] &= ~1u;
      v57.m128i_i32[2] = 0;
      v58.m128i_i32[0] = 0;
      v53.m128i_i32[0] = 1;
      v57.m128i_i64[0] = v21;
      sub_923780(
        v26,
        v7,
        &v44,
        v23,
        v24,
        v25,
        v21,
        0,
        0,
        v53.m128i_i64[0],
        v53.m128i_i64[1],
        v54.m128i_i64[0],
        v54.m128i_i64[1],
        v55.m128i_i64[0],
        v55.m128i_i64[1],
        v56);
      return v44;
    }
    v51 &= ~1u;
    v50 = 0;
    LODWORD(v52) = 0;
    v53.m128i_i32[0] = 1;
    v49 = v21;
    sub_923780(
      v26,
      v7,
      0,
      v23,
      v24,
      v25,
      v21,
      0,
      0,
      v53.m128i_i64[0],
      v53.m128i_i64[1],
      v54.m128i_i64[0],
      v54.m128i_i64[1],
      v55.m128i_i64[0],
      v55.m128i_i64[1],
      v56);
  }
  else
  {
    v47 &= ~1u;
    v46 = 0;
    LODWORD(v48) = 0;
    sub_925900(
      v26,
      v7,
      v22,
      v23,
      v24,
      v25,
      v21,
      0,
      0,
      v53.m128i_i64[0],
      v53.m128i_u64[1],
      v54.m128i_i64[0],
      v54.m128i_i64[1],
      v55.m128i_i64[0],
      v55.m128i_i64[1],
      v56);
  }
  result = 0;
  if ( (*(_BYTE *)(a2 + 25) & 4) == 0 )
  {
    v31 = *a1;
    v32 = _mm_loadu_si128(&v54);
    v33 = _mm_loadu_si128(&v55);
    v53.m128i_i32[0] = v42;
    v34 = _mm_loadu_si128(&v53);
    v58 = v32;
    v60 = v56;
    v57 = v34;
    v59 = v33;
    sub_9286A0(
      (__int64)v45,
      v31,
      v7,
      v27,
      v28,
      v29,
      v34.m128i_i64[0],
      v34.m128i_u64[1],
      v32.m128i_u64[0],
      v32.m128i_i64[1],
      v33.m128i_i64[0],
      v33.m128i_i64[1],
      v56);
    return v45[0];
  }
  return result;
}

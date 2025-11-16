// Function: sub_F94A80
// Address: 0xf94a80
//
__m128i *__fastcall sub_F94A80(__m128i *a1, const __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i v6; // xmm1
  __m128i v7; // xmm0
  void (__fastcall *v8)(_BYTE *, const __m128i *, __int64); // rax
  __int64 v9; // rax
  __m128i v10; // xmm1
  __m128i v11; // xmm0
  void (__fastcall *v12)(_BYTE *, _BYTE *, __int64); // rax
  __m128i v13; // xmm6
  __m128i v14; // xmm7
  __m128i v15; // xmm1
  __m128i v16; // xmm0
  void (__fastcall *v17)(_BYTE *, const __m128i *, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64); // rax
  __int64 v18; // rax
  __m128i v19; // xmm6
  __m128i v20; // xmm7
  void (__fastcall *v21)(_BYTE *, _BYTE *, __int64); // rax
  __m128i v22; // xmm2
  __m128i v23; // xmm3
  __m128i v24; // xmm2
  __m128i v25; // xmm3
  __m128i v26; // xmm1
  __m128i v27; // xmm0
  __int64 v28; // rax
  __m128i v29; // xmm4
  __m128i v30; // xmm5
  void (__fastcall *v31)(_BYTE *, _BYTE *, __int64); // rax
  __m128i v32; // xmm4
  __m128i v33; // xmm5
  void (__fastcall *v34)(__m128i *, _BYTE *, __int64); // rax
  __m128i v36; // xmm6
  __m128i v37; // xmm7
  __m128i v38; // [rsp+0h] [rbp-1A0h] BYREF
  __m128i v39; // [rsp+10h] [rbp-190h] BYREF
  _BYTE v40[16]; // [rsp+20h] [rbp-180h] BYREF
  void (__fastcall *v41)(_BYTE *, _BYTE *, __int64); // [rsp+30h] [rbp-170h]
  __int64 v42; // [rsp+38h] [rbp-168h]
  __m128i v43; // [rsp+40h] [rbp-160h] BYREF
  __m128i v44; // [rsp+50h] [rbp-150h] BYREF
  _BYTE v45[16]; // [rsp+60h] [rbp-140h] BYREF
  void (__fastcall *v46)(_BYTE *, _BYTE *, __int64); // [rsp+70h] [rbp-130h]
  __int64 v47; // [rsp+78h] [rbp-128h]
  __m128i v48; // [rsp+80h] [rbp-120h] BYREF
  __m128i v49; // [rsp+90h] [rbp-110h] BYREF
  _BYTE v50[16]; // [rsp+A0h] [rbp-100h] BYREF
  void (__fastcall *v51)(_BYTE *, _BYTE *, __int64); // [rsp+B0h] [rbp-F0h]
  __int64 v52; // [rsp+B8h] [rbp-E8h]
  __m128i v53; // [rsp+C0h] [rbp-E0h] BYREF
  __m128i v54; // [rsp+D0h] [rbp-D0h] BYREF
  _BYTE v55[16]; // [rsp+E0h] [rbp-C0h] BYREF
  void (__fastcall *v56)(_BYTE *, _BYTE *, __int64); // [rsp+F0h] [rbp-B0h]
  __int64 v57; // [rsp+F8h] [rbp-A8h]
  __m128i v58; // [rsp+100h] [rbp-A0h] BYREF
  __m128i v59; // [rsp+110h] [rbp-90h] BYREF
  _BYTE v60[16]; // [rsp+120h] [rbp-80h] BYREF
  void (__fastcall *v61)(_BYTE *, _BYTE *, __int64); // [rsp+130h] [rbp-70h]
  __int64 v62; // [rsp+138h] [rbp-68h]
  __m128i v63; // [rsp+140h] [rbp-60h] BYREF
  __m128i v64; // [rsp+150h] [rbp-50h] BYREF
  _BYTE v65[16]; // [rsp+160h] [rbp-40h] BYREF
  void (__fastcall *v66)(_BYTE *, _BYTE *, __int64); // [rsp+170h] [rbp-30h]
  __int64 v67; // [rsp+178h] [rbp-28h]

  v6 = _mm_loadu_si128(a2);
  v7 = _mm_loadu_si128(a2 + 1);
  v51 = 0;
  v8 = (void (__fastcall *)(_BYTE *, const __m128i *, __int64))a2[3].m128i_i64[0];
  v48 = v6;
  v49 = v7;
  if ( v8 )
  {
    v8(v50, a2 + 2, 2);
    v9 = a2[3].m128i_i64[1];
    v10 = _mm_loadu_si128(&v48);
    v66 = 0;
    v11 = _mm_loadu_si128(&v49);
    v52 = v9;
    v12 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))a2[3].m128i_i64[0];
    v63 = v10;
    v51 = v12;
    v64 = v11;
    if ( v12 )
    {
      v12(v65, v50, 2);
      v13 = _mm_loadu_si128(&v63);
      v56 = 0;
      v14 = _mm_loadu_si128(&v64);
      v67 = v52;
      v53 = v13;
      v66 = v51;
      v54 = v14;
      if ( v51 )
      {
        v51(v55, v65, 2);
        v57 = v67;
        v56 = v66;
        if ( v66 )
          v66(v65, v65, 3);
      }
    }
    else
    {
      v56 = 0;
      v53 = v10;
      v54 = v11;
    }
  }
  else
  {
    v56 = 0;
    v63 = v6;
    v64 = v7;
    v53 = v6;
    v54 = v7;
  }
  v15 = _mm_loadu_si128(a2 + 4);
  v16 = _mm_loadu_si128(a2 + 5);
  v41 = 0;
  v17 = (void (__fastcall *)(_BYTE *, const __m128i *, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64))a2[7].m128i_i64[0];
  v38 = v15;
  v39 = v16;
  if ( !v17 )
  {
    v63 = v15;
    v64 = v16;
    goto LABEL_34;
  }
  v17(v40, a2 + 6, 2, a4, a5, a6, v38.m128i_i64[0], v38.m128i_i64[1], v39.m128i_i64[0], v39.m128i_i64[1]);
  v18 = a2[7].m128i_i64[1];
  v19 = _mm_loadu_si128(&v38);
  v66 = 0;
  v20 = _mm_loadu_si128(&v39);
  v42 = v18;
  v21 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))a2[7].m128i_i64[0];
  v63 = v19;
  v41 = v21;
  v64 = v20;
  if ( !v21 )
  {
LABEL_34:
    v36 = _mm_loadu_si128(&v63);
    v37 = _mm_loadu_si128(&v64);
    v46 = 0;
    v43 = v36;
    v44 = v37;
    goto LABEL_11;
  }
  v21(v65, v40, 2);
  v22 = _mm_loadu_si128(&v63);
  v46 = 0;
  v23 = _mm_loadu_si128(&v64);
  v67 = v42;
  v43 = v22;
  v66 = v41;
  v44 = v23;
  if ( v41 )
  {
    v41(v45, v65, 2);
    v47 = v67;
    v46 = v66;
    if ( v66 )
      v66(v65, v65, 3);
  }
LABEL_11:
  v24 = _mm_loadu_si128(&v53);
  v25 = _mm_loadu_si128(&v54);
  v66 = 0;
  v63 = v24;
  v64 = v25;
  if ( v56 )
  {
    v56(v65, v55, 2);
    v67 = v57;
    v66 = v56;
  }
  v26 = _mm_loadu_si128(&v43);
  v27 = _mm_loadu_si128(&v44);
  v61 = 0;
  v58 = v26;
  v59 = v27;
  if ( v46 )
  {
    v46(v60, v45, 2);
    v28 = v47;
    v29 = _mm_loadu_si128(&v58);
    a1[3].m128i_i64[0] = 0;
    v30 = _mm_loadu_si128(&v59);
    v62 = v28;
    v31 = v46;
    *a1 = v29;
    v61 = v31;
    a1[1] = v30;
    if ( v31 )
    {
      v31((__m128i *)a1[2].m128i_i8, v60, 2);
      a1[3].m128i_i64[1] = v62;
      a1[3].m128i_i64[0] = (__int64)v61;
    }
  }
  else
  {
    a1[3].m128i_i64[0] = 0;
    *a1 = v26;
    a1[1] = v27;
  }
  v32 = _mm_loadu_si128(&v63);
  v33 = _mm_loadu_si128(&v64);
  a1[7].m128i_i64[0] = 0;
  v34 = (void (__fastcall *)(__m128i *, _BYTE *, __int64))v66;
  a1[4] = v32;
  a1[5] = v33;
  if ( v34 )
  {
    v34(a1 + 6, v65, 2);
    a1[7].m128i_i64[1] = v67;
    a1[7].m128i_i64[0] = (__int64)v66;
  }
  if ( v61 )
    v61(v60, v60, 3);
  if ( v66 )
    v66(v65, v65, 3);
  if ( v46 )
    v46(v45, v45, 3);
  if ( v41 )
    v41(v40, v40, 3);
  if ( v56 )
    v56(v55, v55, 3);
  if ( v51 )
    v51(v50, v50, 3);
  return a1;
}

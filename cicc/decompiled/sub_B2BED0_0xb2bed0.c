// Function: sub_B2BED0
// Address: 0xb2bed0
//
__int64 __fastcall sub_B2BED0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // r12
  __m128i v3; // xmm0
  __m128i v4; // xmm1
  __m128i *v5; // rdi
  __m128i v6; // xmm2
  __m128i v7; // xmm3
  __m128i v8; // xmm4
  __m128i v9; // xmm5
  __m128i v10; // xmm6
  __m128i v11; // xmm7
  void (__fastcall *v12)(_BYTE *, _BYTE *, __int64); // rax
  __int64 v13; // rsi
  int v14; // r12d
  __int64 v15; // rdx
  unsigned int v17; // [rsp+14h] [rbp-23Ch]
  __int64 v18; // [rsp+18h] [rbp-238h]
  __m128i v19; // [rsp+20h] [rbp-230h] BYREF
  __m128i v20; // [rsp+30h] [rbp-220h] BYREF
  _BYTE v21[16]; // [rsp+40h] [rbp-210h] BYREF
  void (__fastcall *v22)(_BYTE *, _BYTE *, __int64); // [rsp+50h] [rbp-200h]
  unsigned __int8 (__fastcall *v23)(_BYTE *, __int64); // [rsp+58h] [rbp-1F8h]
  __m128i v24; // [rsp+60h] [rbp-1F0h] BYREF
  __m128i v25; // [rsp+70h] [rbp-1E0h] BYREF
  _BYTE v26[16]; // [rsp+80h] [rbp-1D0h] BYREF
  void (__fastcall *v27)(_BYTE *, _BYTE *, __int64); // [rsp+90h] [rbp-1C0h]
  __int64 v28; // [rsp+98h] [rbp-1B8h]
  __m128i v29; // [rsp+A0h] [rbp-1B0h]
  __m128i v30; // [rsp+B0h] [rbp-1A0h]
  _BYTE v31[16]; // [rsp+C0h] [rbp-190h] BYREF
  void (__fastcall *v32)(_BYTE *, _BYTE *, __int64); // [rsp+D0h] [rbp-180h]
  unsigned __int8 (__fastcall *v33)(_BYTE *, __int64); // [rsp+D8h] [rbp-178h]
  __m128i v34; // [rsp+E0h] [rbp-170h]
  __m128i v35; // [rsp+F0h] [rbp-160h]
  _BYTE v36[16]; // [rsp+100h] [rbp-150h] BYREF
  void (__fastcall *v37)(_BYTE *, _BYTE *, __int64); // [rsp+110h] [rbp-140h]
  __int64 v38; // [rsp+118h] [rbp-138h]
  __m128i v39; // [rsp+120h] [rbp-130h] BYREF
  __m128i v40; // [rsp+130h] [rbp-120h] BYREF
  _BYTE v41[16]; // [rsp+140h] [rbp-110h] BYREF
  void (__fastcall *v42)(_BYTE *, _BYTE *, __int64); // [rsp+150h] [rbp-100h]
  unsigned __int8 (__fastcall *v43)(_BYTE *, __int64); // [rsp+158h] [rbp-F8h]
  _BYTE v44[16]; // [rsp+180h] [rbp-D0h] BYREF
  void (__fastcall *v45)(_BYTE *, _BYTE *, __int64); // [rsp+190h] [rbp-C0h]
  __m128i v46[2]; // [rsp+1A0h] [rbp-B0h] BYREF
  _BYTE v47[16]; // [rsp+1C0h] [rbp-90h] BYREF
  void (__fastcall *v48)(_BYTE *, _BYTE *, __int64); // [rsp+1D0h] [rbp-80h]
  __m128i v49; // [rsp+1E0h] [rbp-70h] BYREF
  __m128i v50; // [rsp+1F0h] [rbp-60h] BYREF
  _BYTE v51[16]; // [rsp+200h] [rbp-50h] BYREF
  void (__fastcall *v52)(_BYTE *, _BYTE *, __int64); // [rsp+210h] [rbp-40h]
  __int64 v53; // [rsp+218h] [rbp-38h]

  v1 = *(_QWORD *)(a1 + 80);
  v18 = a1 + 72;
  v17 = 0;
  if ( a1 + 72 != v1 )
  {
    while ( 1 )
    {
      v2 = v1 - 24;
      if ( !v1 )
        v2 = 0;
      sub_AA69B0(v46, v2, 1);
      v3 = _mm_loadu_si128(&v49);
      v27 = 0;
      v4 = _mm_loadu_si128(&v50);
      v24 = v3;
      v25 = v4;
      if ( v52 )
      {
        v52(v26, v51, 2);
        v28 = v53;
        v27 = v52;
      }
      v5 = &v39;
      sub_AA69B0(&v39, v2, 1);
      v6 = _mm_loadu_si128(&v39);
      v7 = _mm_loadu_si128(&v40);
      v22 = 0;
      v19 = v6;
      v20 = v7;
      if ( v42 )
      {
        v5 = (__m128i *)v21;
        v42(v21, v41, 2);
        v23 = v43;
        v22 = v42;
      }
      v8 = _mm_loadu_si128(&v24);
      v9 = _mm_loadu_si128(&v25);
      v37 = 0;
      v34 = v8;
      v35 = v9;
      if ( v27 )
      {
        v5 = (__m128i *)v36;
        v27(v36, v26, 2);
        v38 = v28;
        v37 = v27;
      }
      v10 = _mm_loadu_si128(&v19);
      v11 = _mm_loadu_si128(&v20);
      v32 = 0;
      v12 = v22;
      v29 = v10;
      v30 = v11;
      if ( v22 )
        break;
      v13 = v29.m128i_i64[0];
      if ( v34.m128i_i64[0] != v29.m128i_i64[0] )
        goto LABEL_12;
LABEL_25:
      if ( v37 )
        v37(v36, v36, 3);
      if ( v22 )
        v22(v21, v21, 3);
      if ( v45 )
        v45(v44, v44, 3);
      if ( v42 )
        v42(v41, v41, 3);
      if ( v27 )
        v27(v26, v26, 3);
      if ( v52 )
        v52(v51, v51, 3);
      if ( v48 )
        v48(v47, v47, 3);
      v1 = *(_QWORD *)(v1 + 8);
      if ( v18 == v1 )
        return v17;
    }
    v5 = (__m128i *)v31;
    v22(v31, v21, 2);
    v13 = v29.m128i_i64[0];
    v33 = v23;
    v12 = v22;
    v32 = v22;
    if ( v34.m128i_i64[0] != v29.m128i_i64[0] )
    {
LABEL_12:
      v14 = 0;
      do
      {
        v13 = *(_QWORD *)(v13 + 8);
        v29.m128i_i16[4] = 0;
        v29.m128i_i64[0] = v13;
        if ( v30.m128i_i64[0] != v13 )
        {
          while ( 1 )
          {
            v15 = v13 - 24;
            if ( v13 )
              v13 -= 24;
            if ( !v12 )
              sub_4263D6(v5, v13, v15);
            v5 = (__m128i *)v31;
            if ( v33(v31, v13) )
              break;
            v13 = *(_QWORD *)(v29.m128i_i64[0] + 8);
            v29.m128i_i16[4] = 0;
            v12 = v32;
            v29.m128i_i64[0] = v13;
            if ( v30.m128i_i64[0] == v13 )
              goto LABEL_21;
          }
          v13 = v29.m128i_i64[0];
          v12 = v32;
        }
LABEL_21:
        ++v14;
      }
      while ( v34.m128i_i64[0] != v13 );
      v17 += v14;
    }
    if ( v12 )
      v12(v31, v31, 3);
    goto LABEL_25;
  }
  return v17;
}

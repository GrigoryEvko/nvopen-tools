// Function: sub_AA6A60
// Address: 0xaa6a60
//
__int64 __fastcall sub_AA6A60(__int64 a1)
{
  __m128i *v2; // rdi
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __m128i v6; // xmm4
  __m128i v7; // xmm5
  __m128i v8; // xmm6
  __m128i v9; // xmm7
  void (__fastcall *v10)(_BYTE *, _BYTE *, __int64); // rax
  __int64 v11; // rsi
  __int64 v12; // r12
  __int64 v13; // rdx
  __m128i v15; // [rsp+0h] [rbp-210h] BYREF
  __m128i v16; // [rsp+10h] [rbp-200h] BYREF
  _BYTE v17[16]; // [rsp+20h] [rbp-1F0h] BYREF
  void (__fastcall *v18)(_BYTE *, _BYTE *, __int64); // [rsp+30h] [rbp-1E0h]
  unsigned __int8 (__fastcall *v19)(_BYTE *, __int64); // [rsp+38h] [rbp-1D8h]
  __m128i v20; // [rsp+40h] [rbp-1D0h] BYREF
  __m128i v21; // [rsp+50h] [rbp-1C0h] BYREF
  _BYTE v22[16]; // [rsp+60h] [rbp-1B0h] BYREF
  void (__fastcall *v23)(_BYTE *, _BYTE *, __int64); // [rsp+70h] [rbp-1A0h]
  __int64 v24; // [rsp+78h] [rbp-198h]
  __m128i v25; // [rsp+80h] [rbp-190h]
  __m128i v26; // [rsp+90h] [rbp-180h]
  _BYTE v27[16]; // [rsp+A0h] [rbp-170h] BYREF
  void (__fastcall *v28)(_BYTE *, _BYTE *, __int64); // [rsp+B0h] [rbp-160h]
  unsigned __int8 (__fastcall *v29)(_BYTE *, __int64); // [rsp+B8h] [rbp-158h]
  __m128i v30; // [rsp+C0h] [rbp-150h]
  __m128i v31; // [rsp+D0h] [rbp-140h]
  _BYTE v32[16]; // [rsp+E0h] [rbp-130h] BYREF
  void (__fastcall *v33)(_BYTE *, _BYTE *, __int64); // [rsp+F0h] [rbp-120h]
  __int64 v34; // [rsp+F8h] [rbp-118h]
  __m128i v35; // [rsp+100h] [rbp-110h] BYREF
  __m128i v36; // [rsp+110h] [rbp-100h] BYREF
  _BYTE v37[16]; // [rsp+120h] [rbp-F0h] BYREF
  void (__fastcall *v38)(_BYTE *, _BYTE *, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64); // [rsp+130h] [rbp-E0h]
  unsigned __int8 (__fastcall *v39)(_BYTE *, __int64); // [rsp+138h] [rbp-D8h]
  _BYTE v40[16]; // [rsp+160h] [rbp-B0h] BYREF
  void (__fastcall *v41)(_BYTE *, _BYTE *, __int64); // [rsp+170h] [rbp-A0h]
  __m128i v42[2]; // [rsp+180h] [rbp-90h] BYREF
  _BYTE v43[16]; // [rsp+1A0h] [rbp-70h] BYREF
  void (__fastcall *v44)(_BYTE *, _BYTE *, __int64); // [rsp+1B0h] [rbp-60h]
  __m128i v45; // [rsp+1C0h] [rbp-50h] BYREF
  __m128i v46; // [rsp+1D0h] [rbp-40h] BYREF
  _BYTE v47[16]; // [rsp+1E0h] [rbp-30h] BYREF
  void (__fastcall *v48)(_BYTE *, _BYTE *, __int64); // [rsp+1F0h] [rbp-20h]
  __int64 v49; // [rsp+1F8h] [rbp-18h]

  sub_AA69B0(v42, a1, 1);
  v23 = 0;
  v20 = _mm_loadu_si128(&v45);
  v21 = _mm_loadu_si128(&v46);
  if ( v48 )
  {
    v48(v22, v47, 2);
    v24 = v49;
    v23 = v48;
  }
  v2 = &v35;
  sub_AA69B0(&v35, a1, 1);
  v18 = 0;
  v15 = _mm_loadu_si128(&v35);
  v16 = _mm_loadu_si128(&v36);
  if ( v38 )
  {
    v2 = (__m128i *)v17;
    v38(v17, v37, 2, v3, v4, v5, v15.m128i_i64[0], v15.m128i_i64[1], v16.m128i_i64[0], v16.m128i_i64[1]);
    v19 = v39;
    v18 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))v38;
  }
  v6 = _mm_loadu_si128(&v20);
  v7 = _mm_loadu_si128(&v21);
  v33 = 0;
  v30 = v6;
  v31 = v7;
  if ( v23 )
  {
    v2 = (__m128i *)v32;
    v23(v32, v22, 2);
    v34 = v24;
    v33 = v23;
  }
  v8 = _mm_loadu_si128(&v15);
  v9 = _mm_loadu_si128(&v16);
  v28 = 0;
  v10 = v18;
  v25 = v8;
  v26 = v9;
  if ( !v18 )
  {
    v11 = v25.m128i_i64[0];
    v12 = 0;
    if ( v30.m128i_i64[0] == v25.m128i_i64[0] )
      goto LABEL_21;
    goto LABEL_9;
  }
  v2 = (__m128i *)v27;
  v18(v27, v17, 2);
  v11 = v25.m128i_i64[0];
  v29 = v19;
  v10 = v18;
  v28 = v18;
  if ( v25.m128i_i64[0] != v30.m128i_i64[0] )
  {
LABEL_9:
    v12 = 0;
    do
    {
      v11 = *(_QWORD *)(v11 + 8);
      v25.m128i_i16[4] = 0;
      v25.m128i_i64[0] = v11;
      if ( v26.m128i_i64[0] != v11 )
      {
        while ( 1 )
        {
          v13 = v11 - 24;
          if ( v11 )
            v11 -= 24;
          if ( !v10 )
            sub_4263D6(v2, v11, v13);
          v2 = (__m128i *)v27;
          if ( v29(v27, v11) )
            break;
          v11 = *(_QWORD *)(v25.m128i_i64[0] + 8);
          v25.m128i_i16[4] = 0;
          v10 = v28;
          v25.m128i_i64[0] = v11;
          if ( v26.m128i_i64[0] == v11 )
            goto LABEL_18;
        }
        v11 = v25.m128i_i64[0];
        v10 = v28;
      }
LABEL_18:
      ++v12;
    }
    while ( v30.m128i_i64[0] != v11 );
    goto LABEL_19;
  }
  v12 = 0;
LABEL_19:
  if ( v10 )
    v10(v27, v27, 3);
LABEL_21:
  if ( v33 )
    v33(v32, v32, 3);
  if ( v18 )
    v18(v17, v17, 3);
  if ( v41 )
    v41(v40, v40, 3);
  if ( v38 )
    ((void (__fastcall *)(_BYTE *, _BYTE *, __int64))v38)(v37, v37, 3);
  if ( v23 )
    v23(v22, v22, 3);
  if ( v48 )
    v48(v47, v47, 3);
  if ( v44 )
    v44(v43, v43, 3);
  return v12;
}

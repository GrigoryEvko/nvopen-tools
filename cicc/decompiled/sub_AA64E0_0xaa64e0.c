// Function: sub_AA64E0
// Address: 0xaa64e0
//
__m128i *__fastcall sub_AA64E0(__m128i *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  void (__fastcall *v5)(_BYTE *, __int64, __int64); // rax
  __int64 v6; // rbx
  __int64 v7; // rax
  void (__fastcall *v8)(__m128i *, _BYTE *, __int64); // rax
  __m128i *v9; // rdi
  __int64 v10; // rsi
  void (__fastcall *v11)(__m128i *, __m128i *, __int64); // rax
  void (__fastcall *v12)(_BYTE *, __int64, __int64); // rax
  __int64 v13; // rax
  void (__fastcall *v14)(__m128i *, _BYTE *, __int64); // rax
  __int64 v15; // rax
  __m128i v16; // xmm0
  __m128i v17; // xmm2
  __m128i v18; // xmm1
  __m128i v19; // xmm6
  __int64 v20; // rdx
  __m128i v21; // xmm5
  __m128i v22; // xmm4
  __m128i v23; // xmm3
  void (__fastcall *v24)(_BYTE *, _BYTE *, __int64); // rax
  __int64 v26; // [rsp+8h] [rbp-178h]
  _BYTE v27[16]; // [rsp+10h] [rbp-170h] BYREF
  __m128i v28; // [rsp+20h] [rbp-160h]
  _BYTE v29[16]; // [rsp+30h] [rbp-150h] BYREF
  __m128i v30; // [rsp+40h] [rbp-140h]
  __m128i v31; // [rsp+50h] [rbp-130h] BYREF
  __m128i v32; // [rsp+60h] [rbp-120h] BYREF
  __m128i v33; // [rsp+70h] [rbp-110h] BYREF
  __m128i v34; // [rsp+80h] [rbp-100h]
  __m128i v35; // [rsp+90h] [rbp-F0h] BYREF
  __m128i v36; // [rsp+A0h] [rbp-E0h] BYREF
  __m128i v37; // [rsp+B0h] [rbp-D0h] BYREF
  __m128i v38; // [rsp+C0h] [rbp-C0h]
  __m128i v39; // [rsp+D0h] [rbp-B0h] BYREF
  __m128i v40; // [rsp+E0h] [rbp-A0h]
  __m128i v41; // [rsp+F0h] [rbp-90h]
  __m128i v42; // [rsp+110h] [rbp-70h] BYREF
  __m128i v43; // [rsp+120h] [rbp-60h]
  __m128i v44[5]; // [rsp+130h] [rbp-50h] BYREF

  v3 = a3;
  v26 = a2 + 48;
  v5 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a3 + 16);
  v30.m128i_i64[0] = 0;
  v6 = *(_QWORD *)(a2 + 56);
  if ( v5
    && (v5(v29, a3, 2),
        v7 = *(_QWORD *)(v3 + 24),
        v43.m128i_i64[0] = 0,
        v30.m128i_i64[1] = v7,
        v8 = *(void (__fastcall **)(__m128i *, _BYTE *, __int64))(v3 + 16),
        (v30.m128i_i64[0] = (__int64)v8) != 0) )
  {
    v8(&v42, v29, 2);
    v35.m128i_i16[4] = 0;
    v35.m128i_i64[0] = a2 + 48;
    v43 = v30;
    v36.m128i_i64[0] = a2 + 48;
    v36.m128i_i16[4] = 0;
    v38.m128i_i64[0] = 0;
    if ( v30.m128i_i64[0] )
    {
      v9 = &v37;
      ((void (__fastcall *)(__m128i *, __m128i *, __int64))v30.m128i_i64[0])(&v37, &v42, 2);
      v10 = v35.m128i_i64[0];
      v11 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v43.m128i_i64[0];
      v38 = v43;
      if ( v36.m128i_i64[0] != v35.m128i_i64[0] )
      {
        while ( 1 )
        {
          a3 = v10 - 24;
          if ( v10 )
            v10 -= 24;
          if ( !v11 )
            goto LABEL_37;
          v9 = &v37;
          if ( ((unsigned __int8 (__fastcall *)(__m128i *, __int64))v38.m128i_i64[1])(&v37, v10) )
            break;
          v10 = *(_QWORD *)(v35.m128i_i64[0] + 8);
          v35.m128i_i16[4] = 0;
          v35.m128i_i64[0] = v10;
          if ( v36.m128i_i64[0] == v10 )
            break;
          v11 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v38.m128i_i64[0];
        }
        v11 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v43.m128i_i64[0];
      }
      if ( v11 )
        v11(&v42, &v42, 3);
    }
  }
  else
  {
    v38.m128i_i64[0] = 0;
    v35.m128i_i16[4] = 0;
    v35.m128i_i64[0] = a2 + 48;
    v36.m128i_i64[0] = a2 + 48;
    v36.m128i_i16[4] = 0;
  }
  v28.m128i_i64[0] = 0;
  v12 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(v3 + 16);
  if ( !v12 )
  {
    v40.m128i_i64[0] = 0;
    goto LABEL_36;
  }
  v12(v27, v3, 2);
  v13 = *(_QWORD *)(v3 + 24);
  v40.m128i_i64[0] = 0;
  v28.m128i_i64[1] = v13;
  v14 = *(void (__fastcall **)(__m128i *, _BYTE *, __int64))(v3 + 16);
  v28.m128i_i64[0] = (__int64)v14;
  if ( !v14 )
  {
LABEL_36:
    v10 = 1;
    v9 = 0;
    v31.m128i_i64[0] = v6;
    v31.m128i_i16[4] = 1;
    v32.m128i_i64[0] = v26;
    v32.m128i_i16[4] = 0;
    v34.m128i_i64[0] = 0;
    goto LABEL_19;
  }
  v10 = (__int64)v27;
  v9 = &v39;
  v14(&v39, v27, 2);
  a3 = 1;
  v31.m128i_i64[0] = v6;
  v32.m128i_i64[0] = v26;
  v40 = v28;
  v31.m128i_i16[4] = 1;
  v32.m128i_i16[4] = 0;
  v34.m128i_i64[0] = 0;
  if ( v28.m128i_i64[0] )
  {
    v9 = &v33;
    v10 = (__int64)&v39;
    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v28.m128i_i64[0])(&v33, &v39, 2);
    v6 = v31.m128i_i64[0];
    v34 = v40;
    v26 = v32.m128i_i64[0];
  }
LABEL_19:
  if ( v26 != v6 )
  {
    while ( 1 )
    {
      if ( v6 )
        v6 -= 24;
      if ( !v34.m128i_i64[0] )
        break;
      v10 = v6;
      v9 = &v33;
      if ( !((unsigned __int8 (__fastcall *)(__m128i *, __int64))v34.m128i_i64[1])(&v33, v6) )
      {
        v6 = *(_QWORD *)(v31.m128i_i64[0] + 8);
        v31.m128i_i16[4] = 0;
        v31.m128i_i64[0] = v6;
        if ( v32.m128i_i64[0] != v6 )
          continue;
      }
      goto LABEL_26;
    }
LABEL_37:
    sub_4263D6(v9, v10, a3);
  }
LABEL_26:
  if ( v40.m128i_i64[0] )
    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v40.m128i_i64[0])(&v39, &v39, 3);
  v15 = v38.m128i_i64[1];
  v38.m128i_i64[1] = 0;
  v16 = _mm_loadu_si128(&v37);
  v17 = _mm_loadu_si128(&v35);
  v18 = _mm_loadu_si128(&v36);
  v19 = _mm_loadu_si128(v44);
  a1[3].m128i_i64[0] = v34.m128i_i64[0];
  v20 = v38.m128i_i64[0];
  v21 = _mm_loadu_si128(&v31);
  a1[7].m128i_i64[1] = v15;
  v22 = _mm_loadu_si128(&v32);
  v23 = _mm_loadu_si128(&v33);
  v42 = v17;
  v24 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))v28.m128i_i64[0];
  v43 = v18;
  v38.m128i_i64[0] = 0;
  a1[3].m128i_i64[1] = v34.m128i_i64[1];
  a1[7].m128i_i64[0] = v20;
  v37 = v19;
  v44[0] = v16;
  v39 = v21;
  v40 = v22;
  v41 = v23;
  *a1 = v21;
  a1[1] = v22;
  a1[2] = v23;
  a1[4] = v17;
  a1[5] = v18;
  a1[6] = v16;
  if ( v24 )
  {
    v24(v27, v27, 3);
    if ( v38.m128i_i64[0] )
      ((void (__fastcall *)(__m128i *, __m128i *, __int64))v38.m128i_i64[0])(&v37, &v37, 3);
  }
  if ( v30.m128i_i64[0] )
    ((void (__fastcall *)(_BYTE *, _BYTE *, __int64))v30.m128i_i64[0])(v29, v29, 3);
  return a1;
}

// Function: sub_344D5D0
// Address: 0x344d5d0
//
const __m128i *__fastcall sub_344D5D0(__m128i *a1, __m128i *a2)
{
  __m128i *v2; // r12
  void (__fastcall *v3)(_BYTE *, __m128i *, __int64); // rax
  __int64 v4; // rax
  __int64 v5; // r13
  void (__fastcall *v6)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v7; // r13
  const __m128i *v8; // rbx
  const __m128i *v9; // r15
  unsigned __int8 (__fastcall *v10)(__m128i *, __m128i *); // rcx
  void (__fastcall *v11)(_QWORD, _QWORD, _QWORD); // rax
  __m128i v12; // xmm5
  __m128i v13; // xmm0
  unsigned __int8 (__fastcall *v14)(__m128i *, __m128i *); // rdx
  unsigned __int8 (__fastcall *v15)(__m128i *, __m128i *); // rcx
  __m128i v16; // xmm0
  __m128i v17; // xmm6
  __int64 v18; // rdx
  __int64 v19; // r13
  __int64 v20; // rdx
  const __m128i *v21; // r13
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  bool v26; // zf
  __m128i v27; // [rsp+0h] [rbp-150h] BYREF
  __m128i v28; // [rsp+10h] [rbp-140h] BYREF
  __m128i v29; // [rsp+20h] [rbp-130h] BYREF
  __m128i v30; // [rsp+30h] [rbp-120h] BYREF
  __m128i v31; // [rsp+40h] [rbp-110h] BYREF
  __m128i v32; // [rsp+50h] [rbp-100h] BYREF
  _BYTE v33[16]; // [rsp+60h] [rbp-F0h] BYREF
  void (__fastcall *v34)(_QWORD, _QWORD, _QWORD); // [rsp+70h] [rbp-E0h]
  __int64 v35; // [rsp+78h] [rbp-D8h]
  __m128i v36; // [rsp+80h] [rbp-D0h] BYREF
  void (__fastcall *v37)(__m128i *, __m128i *, __int64); // [rsp+90h] [rbp-C0h]
  unsigned __int8 (__fastcall *v38)(_QWORD, _QWORD); // [rsp+98h] [rbp-B8h]
  __m128i v39; // [rsp+A0h] [rbp-B0h] BYREF
  void (__fastcall *v40)(_QWORD, _QWORD, _QWORD); // [rsp+B0h] [rbp-A0h]
  unsigned __int8 (__fastcall *v41)(__m128i *, __m128i *); // [rsp+B8h] [rbp-98h]
  __m128i v42; // [rsp+C0h] [rbp-90h] BYREF
  void (__fastcall *v43)(__m128i *, __m128i *, __int64); // [rsp+D0h] [rbp-80h]
  unsigned __int8 (__fastcall *v44)(__m128i *, __m128i *); // [rsp+D8h] [rbp-78h]
  __m128i v45; // [rsp+E0h] [rbp-70h] BYREF
  void (__fastcall *v46)(_QWORD, _QWORD, _QWORD); // [rsp+F0h] [rbp-60h]
  unsigned __int8 (__fastcall *v47)(__m128i *, __m128i *); // [rsp+F8h] [rbp-58h]
  __m128i v48; // [rsp+100h] [rbp-50h] BYREF
  unsigned __int8 (__fastcall *v49)(_QWORD, _QWORD); // [rsp+118h] [rbp-38h]

  v2 = a1;
  v3 = (void (__fastcall *)(_BYTE *, __m128i *, __int64))a2[1].m128i_i64[0];
  v34 = 0;
  if ( v3 )
  {
    a1 = (__m128i *)v33;
    v3(v33, a2, 2);
    v4 = a2[1].m128i_i64[1];
    v5 = v2->m128i_i64[1];
    v37 = 0;
    v35 = v4;
    v6 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))a2[1].m128i_i64[0];
    v7 = 16 * v5;
    v8 = (const __m128i *)v2->m128i_i64[0];
    v34 = v6;
    v9 = &v8[(unsigned __int64)v7 / 0x10];
    if ( v6 )
    {
      a1 = &v36;
      a2 = (__m128i *)v33;
      v6(&v36, v33, 2);
      v10 = (unsigned __int8 (__fastcall *)(_QWORD, _QWORD))v35;
      v11 = v34;
      goto LABEL_4;
    }
  }
  else
  {
    v8 = (const __m128i *)a1->m128i_i64[0];
    v7 = 16 * a1->m128i_i64[1];
    v9 = (const __m128i *)(a1->m128i_i64[0] + v7);
  }
  v10 = v38;
  v11 = 0;
LABEL_4:
  v12 = _mm_loadu_si128(&v48);
  v13 = _mm_loadu_si128(&v36);
  v41 = (unsigned __int8 (__fastcall *)(__m128i *, __m128i *))v10;
  v14 = v49;
  v40 = v11;
  v37 = 0;
  v15 = v44;
  v38 = v49;
  v43 = 0;
  v36 = v12;
  v48 = v13;
  v39 = v13;
  if ( v11 )
  {
    a2 = &v39;
    a1 = &v42;
    ((void (__fastcall *)(__m128i *, __m128i *, __int64, unsigned __int8 (__fastcall *)(__m128i *, __m128i *)))v11)(
      &v42,
      &v39,
      2,
      v44);
    v15 = v41;
    v11 = v40;
    v14 = v49;
  }
  v16 = _mm_loadu_si128(&v42);
  v17 = _mm_loadu_si128(&v48);
  v44 = (unsigned __int8 (__fastcall *)(__m128i *, __m128i *))v14;
  v18 = v7;
  v19 = v7 >> 6;
  v43 = 0;
  v20 = v18 >> 4;
  v46 = v11;
  v47 = v15;
  v42 = v17;
  v48 = v16;
  v45 = v16;
  if ( v19 <= 0 )
  {
LABEL_32:
    if ( v20 != 2 )
    {
      if ( v20 != 3 )
      {
        if ( v20 != 1 )
          goto LABEL_18;
LABEL_35:
        v48 = _mm_loadu_si128(v8);
        if ( v11 )
        {
          v26 = v47(&v45, &v48) == 0;
          v11 = v46;
          if ( v26 )
            v9 = v8;
          goto LABEL_18;
        }
LABEL_47:
        sub_4263D6(a1, a2, v20);
      }
      v31 = _mm_loadu_si128(v8);
      if ( !v11 )
        goto LABEL_47;
      a2 = &v31;
      a1 = &v45;
      if ( !v47(&v45, &v31) )
      {
LABEL_17:
        v11 = v46;
        v9 = v8;
        goto LABEL_18;
      }
      v11 = v46;
      ++v8;
    }
    v32 = _mm_loadu_si128(v8);
    if ( !v11 )
      goto LABEL_47;
    a2 = &v32;
    a1 = &v45;
    if ( v47(&v45, &v32) )
    {
      v11 = v46;
      ++v8;
      goto LABEL_35;
    }
    goto LABEL_17;
  }
  v21 = &v8[4 * v19];
  while ( 1 )
  {
    v27 = _mm_loadu_si128(v8);
    if ( !v11 )
      goto LABEL_47;
    a2 = &v27;
    a1 = &v45;
    if ( !((unsigned __int8 (__fastcall *)(__m128i *, __m128i *, __int64))v47)(&v45, &v27, v20) )
      goto LABEL_17;
    v28 = _mm_loadu_si128(v8 + 1);
    if ( !v46 )
      goto LABEL_47;
    a2 = &v28;
    a1 = &v45;
    if ( !((unsigned __int8 (__fastcall *)(__m128i *, __m128i *, __int64, __int64, __int64, __int64, __int64, __int64))v47)(
            &v45,
            &v28,
            v20,
            v22,
            v23,
            v24,
            v27.m128i_i64[0],
            v27.m128i_i64[1]) )
    {
      v11 = v46;
      v9 = v8 + 1;
      goto LABEL_18;
    }
    v29 = _mm_loadu_si128(v8 + 2);
    if ( !v46 )
      goto LABEL_47;
    a2 = &v29;
    a1 = &v45;
    if ( !v47(&v45, &v29) )
    {
      v11 = v46;
      v9 = v8 + 2;
      goto LABEL_18;
    }
    v30 = _mm_loadu_si128(v8 + 3);
    if ( !v46 )
      goto LABEL_47;
    a2 = &v30;
    a1 = &v45;
    if ( !v47(&v45, &v30) )
      break;
    v8 += 4;
    v11 = v46;
    if ( v8 == v21 )
    {
      v20 = v9 - v8;
      goto LABEL_32;
    }
  }
  v11 = v46;
  v9 = v8 + 3;
LABEL_18:
  if ( v11 )
    v11(&v45, &v45, 3);
  if ( v43 )
    v43(&v42, &v42, 3);
  if ( v40 )
    v40(&v39, &v39, 3);
  if ( v37 )
    v37(&v36, &v36, 3);
  if ( v34 )
    v34(v33, v33, 3);
  return v9;
}

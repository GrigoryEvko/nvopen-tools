// Function: sub_2F843A0
// Address: 0x2f843a0
//
__m128i *__fastcall sub_2F843A0(__m128i *a1, __int64 *a2, __int64 *a3)
{
  __int64 *v3; // r14
  __m128i *v4; // r12
  __int64 *v5; // r15
  __int64 v6; // rax
  void (__fastcall *v7)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v8; // rax
  void (__fastcall *v9)(_QWORD, _QWORD, _QWORD); // rax
  void (__fastcall *v10)(_QWORD, _QWORD, _QWORD); // rcx
  __int64 *v11; // rax
  void (__fastcall *v12)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v13; // rax
  void (__fastcall *v14)(_QWORD, _QWORD, _QWORD); // rax
  void (__fastcall *v15)(_QWORD, _QWORD, _QWORD); // rcx
  __int64 *v16; // rax
  __int64 v17; // rdi
  __int64 *v18; // rax
  __m128i v19; // xmm2
  __int64 *v20; // rdx
  __m128i v21; // xmm0
  __int64 v22; // rsi
  __int64 *v23; // rax
  __int64 v24; // rcx
  __m128i v25; // xmm1
  __int64 v26; // rdx
  void (__fastcall *v27)(_QWORD, _QWORD, _QWORD); // rax
  __int64 *v29; // [rsp+8h] [rbp-148h]
  __int64 v30; // [rsp+10h] [rbp-140h] BYREF
  __int64 v31; // [rsp+18h] [rbp-138h] BYREF
  _BYTE v32[16]; // [rsp+20h] [rbp-130h] BYREF
  __m128i v33; // [rsp+30h] [rbp-120h]
  _BYTE v34[16]; // [rsp+40h] [rbp-110h] BYREF
  __m128i v35; // [rsp+50h] [rbp-100h]
  __int64 *v36; // [rsp+60h] [rbp-F0h]
  __int64 *v37; // [rsp+68h] [rbp-E8h]
  __m128i v38; // [rsp+70h] [rbp-E0h] BYREF
  __m128i v39; // [rsp+80h] [rbp-D0h]
  __int64 *v40; // [rsp+90h] [rbp-C0h]
  __int64 *v41; // [rsp+98h] [rbp-B8h]
  __m128i v42; // [rsp+A0h] [rbp-B0h] BYREF
  __m128i v43; // [rsp+B0h] [rbp-A0h]
  _QWORD v44[2]; // [rsp+C0h] [rbp-90h] BYREF
  __m128i v45; // [rsp+D0h] [rbp-80h]
  _QWORD v46[2]; // [rsp+F0h] [rbp-60h] BYREF
  __m128i v47; // [rsp+100h] [rbp-50h] BYREF

  v3 = a3;
  v4 = a1;
  v5 = (__int64 *)*a2;
  v6 = *((unsigned int *)a2 + 2);
  v35.m128i_i64[0] = 0;
  v29 = &v5[v6];
  v7 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))a3[2];
  if ( !v7 )
    goto LABEL_32;
  a2 = a3;
  a1 = (__m128i *)v34;
  v7(v34, a3, 2);
  v8 = v3[3];
  v47.m128i_i64[0] = 0;
  v35.m128i_i64[1] = v8;
  v9 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v3[2];
  v35.m128i_i64[0] = (__int64)v9;
  if ( v9 )
  {
    a2 = (__int64 *)v34;
    a1 = (__m128i *)v46;
    v9(v46, v34, 2);
    a3 = v29;
    v43.m128i_i64[0] = 0;
    v40 = v29;
    v47 = v35;
    v41 = v29;
    if ( v35.m128i_i64[0] )
    {
      a2 = v46;
      a1 = &v42;
      ((void (__fastcall *)(__m128i *, _QWORD *, __int64))v35.m128i_i64[0])(&v42, v46, 2);
      v10 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v47.m128i_i64[0];
      v11 = v40;
      v43 = v47;
      if ( v40 != v41 )
      {
        while ( 1 )
        {
          v30 = *v11;
          if ( !v10 )
            goto LABEL_37;
          a2 = &v30;
          a1 = &v42;
          if ( ((unsigned __int8 (__fastcall *)(__m128i *, __int64 *))v43.m128i_i64[1])(&v42, &v30) )
            break;
          v11 = v40 + 1;
          v40 = v11;
          if ( v41 == v11 )
            break;
          v10 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v43.m128i_i64[0];
        }
        v10 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v47.m128i_i64[0];
      }
      if ( v10 )
      {
        a2 = v46;
        a1 = (__m128i *)v46;
        v10(v46, v46, 3);
      }
    }
  }
  else
  {
LABEL_32:
    v43.m128i_i64[0] = 0;
    v40 = v29;
    v41 = v29;
  }
  v33.m128i_i64[0] = 0;
  v12 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v3[2];
  if ( !v12 )
  {
    v45.m128i_i64[0] = 0;
    goto LABEL_34;
  }
  a2 = v3;
  a1 = (__m128i *)v32;
  v12(v32, v3, 2);
  v13 = v3[3];
  v45.m128i_i64[0] = 0;
  v33.m128i_i64[1] = v13;
  v14 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v3[2];
  v33.m128i_i64[0] = (__int64)v14;
  if ( !v14 )
  {
LABEL_34:
    v36 = v5;
    v39.m128i_i64[0] = 0;
    v37 = v29;
    goto LABEL_35;
  }
  a2 = (__int64 *)v32;
  a1 = (__m128i *)v44;
  v14(v44, v32, 2);
  a3 = v29;
  v36 = v5;
  v39.m128i_i64[0] = 0;
  v37 = v29;
  v45 = v33;
  if ( v33.m128i_i64[0] )
  {
    a2 = v44;
    a1 = &v38;
    ((void (__fastcall *)(__m128i *, _QWORD *, __int64))v33.m128i_i64[0])(&v38, v44, 2);
    v15 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v45.m128i_i64[0];
    v16 = v36;
    v39 = v45;
    if ( v36 != v37 )
    {
      while ( 1 )
      {
        v31 = *v16;
        if ( !v15 )
          goto LABEL_37;
        a2 = &v31;
        a1 = &v38;
        if ( ((unsigned __int8 (__fastcall *)(__m128i *, __int64 *))v39.m128i_i64[1])(&v38, &v31) )
          break;
        v16 = v36 + 1;
        v36 = v16;
        if ( v37 == v16 )
          break;
        v15 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v39.m128i_i64[0];
      }
      v15 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v45.m128i_i64[0];
    }
    if ( v15 )
      v15(v44, v44, 3);
    v5 = v37;
    v17 = v39.m128i_i64[0];
    goto LABEL_26;
  }
LABEL_35:
  if ( v29 != v5 )
  {
    v31 = *v5;
LABEL_37:
    sub_4263D6(a1, a2, a3);
  }
  v17 = 0;
LABEL_26:
  v18 = v40;
  v19 = _mm_loadu_si128(&v47);
  v4->m128i_i64[1] = (__int64)v5;
  v20 = v36;
  v21 = _mm_loadu_si128(&v42);
  v4[2].m128i_i64[0] = v17;
  v46[0] = v18;
  v22 = v43.m128i_i64[0];
  v4[3].m128i_i64[0] = (__int64)v18;
  v23 = v41;
  v24 = v43.m128i_i64[1];
  v25 = _mm_loadu_si128(&v38);
  v44[0] = v20;
  v4->m128i_i64[0] = (__int64)v20;
  v26 = v39.m128i_i64[1];
  v4[3].m128i_i64[1] = (__int64)v23;
  v27 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v33.m128i_i64[0];
  v43 = 0u;
  v4[2].m128i_i64[1] = v26;
  v4[5].m128i_i64[0] = v22;
  v4[5].m128i_i64[1] = v24;
  v42 = v19;
  v47 = v21;
  v45 = v25;
  v4[1] = v25;
  v4[4] = v21;
  if ( v27 )
  {
    v27(v32, v32, 3);
    if ( v43.m128i_i64[0] )
      ((void (__fastcall *)(__m128i *, __m128i *, __int64))v43.m128i_i64[0])(&v42, &v42, 3);
  }
  if ( v35.m128i_i64[0] )
    ((void (__fastcall *)(_BYTE *, _BYTE *, __int64))v35.m128i_i64[0])(v34, v34, 3);
  return v4;
}

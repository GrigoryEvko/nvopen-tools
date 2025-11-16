// Function: sub_25BDD80
// Address: 0x25bdd80
//
__int64 *__fastcall sub_25BDD80(__int64 *a1, __m128i *a2, __m128i *a3)
{
  __m128i *v3; // r13
  __m128i *v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // r12
  __m128i *v8; // r14
  __int64 v9; // rdi
  __int64 v10; // rax
  __m128i v11; // xmm1
  __m128i v12; // xmm0
  __int64 v13; // rdx
  __m128i v14; // xmm2
  void (__fastcall *v15)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rax
  __m128i v19; // xmm3
  __m128i v20; // xmm0
  __int64 v21; // rdx
  __m128i v22; // xmm4
  void (__fastcall *v23)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rax
  __m128i v27; // xmm5
  __m128i v28; // xmm0
  __int64 v29; // rdx
  __m128i v30; // xmm6
  void (__fastcall *v31)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v32; // rdi
  __int32 v33; // eax
  __int64 v34; // rcx
  void (__fastcall *v35)(__int64, __int64, __int64); // rax
  void (__fastcall *v36)(__int64, __int64, __int64); // rax
  void (__fastcall *v37)(__int64, __int64, __int64); // rax
  __int64 v39; // [rsp+0h] [rbp-60h]
  __m128i v41; // [rsp+10h] [rbp-50h] BYREF
  void (__fastcall *v42)(_QWORD, _QWORD, _QWORD); // [rsp+20h] [rbp-40h]
  __int64 v43; // [rsp+28h] [rbp-38h]

  v3 = a2;
  v4 = a3;
  v5 = *a1;
  v6 = *a1 + 104LL * *((unsigned int *)a1 + 2);
  v39 = v6 - (_QWORD)a3;
  v7 = 0x4EC4EC4EC4EC4EC5LL * ((v6 - (__int64)a3) >> 3);
  if ( v6 - (__int64)a3 > 0 )
  {
    v8 = a2;
    do
    {
      v9 = v4[1].m128i_i64[0];
      v10 = v43;
      v4[1].m128i_i64[0] = 0;
      v11 = _mm_loadu_si128(&v41);
      v12 = _mm_loadu_si128(v4);
      v13 = v4[1].m128i_i64[1];
      v4[1].m128i_i64[1] = v10;
      *v4 = v11;
      v14 = _mm_loadu_si128(v8);
      v15 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v8[1].m128i_i64[0];
      v8[1].m128i_i64[0] = v9;
      v16 = v8[1].m128i_i64[1];
      v42 = v15;
      v43 = v16;
      v8[1].m128i_i64[1] = v13;
      v41 = v14;
      *v8 = v12;
      if ( v15 )
        v15(&v41, &v41, 3);
      v17 = v4[3].m128i_i64[0];
      v18 = v43;
      v4[3].m128i_i64[0] = 0;
      v19 = _mm_loadu_si128(&v41);
      v20 = _mm_loadu_si128(v4 + 2);
      v21 = v4[3].m128i_i64[1];
      v4[3].m128i_i64[1] = v18;
      v4[2] = v19;
      v22 = _mm_loadu_si128(v8 + 2);
      v23 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v8[3].m128i_i64[0];
      v8[3].m128i_i64[0] = v17;
      v24 = v8[3].m128i_i64[1];
      v42 = v23;
      v43 = v24;
      v8[3].m128i_i64[1] = v21;
      v41 = v22;
      v8[2] = v20;
      if ( v23 )
        v23(&v41, &v41, 3);
      v25 = v4[5].m128i_i64[0];
      v26 = v43;
      v4[5].m128i_i64[0] = 0;
      v27 = _mm_loadu_si128(&v41);
      v28 = _mm_loadu_si128(v4 + 4);
      v29 = v4[5].m128i_i64[1];
      v4[5].m128i_i64[1] = v26;
      v4[4] = v27;
      v30 = _mm_loadu_si128(v8 + 4);
      v31 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v8[5].m128i_i64[0];
      v8[5].m128i_i64[0] = v25;
      v32 = v8[5].m128i_i64[1];
      v42 = v31;
      v43 = v32;
      v8[5].m128i_i64[1] = v29;
      v41 = v30;
      v8[4] = v28;
      if ( v31 )
        v31(&v41, &v41, 3);
      v33 = v4[6].m128i_i32[0];
      v8 = (__m128i *)((char *)v8 + 104);
      v4 = (__m128i *)((char *)v4 + 104);
      v8[-1].m128i_i32[2] = v33;
      v8[-1].m128i_i8[12] = v4[-1].m128i_i8[12];
      --v7;
    }
    while ( v7 );
    v34 = v39;
    if ( v39 <= 0 )
      v34 = 104;
    v3 = (__m128i *)((char *)a2 + v34);
    v5 = *a1;
    v6 = *a1 + 104LL * *((unsigned int *)a1 + 2);
  }
  if ( (__m128i *)v6 != v3 )
  {
    do
    {
      v35 = *(void (__fastcall **)(__int64, __int64, __int64))(v6 - 24);
      v6 -= 104;
      if ( v35 )
        v35(v6 + 64, v6 + 64, 3);
      v36 = *(void (__fastcall **)(__int64, __int64, __int64))(v6 + 48);
      if ( v36 )
        v36(v6 + 32, v6 + 32, 3);
      v37 = *(void (__fastcall **)(__int64, __int64, __int64))(v6 + 16);
      if ( v37 )
        v37(v6, v6, 3);
    }
    while ( (__m128i *)v6 != v3 );
    v5 = *a1;
  }
  *((_DWORD *)a1 + 2) = -991146299 * (((__int64)v3->m128i_i64 - v5) >> 3);
  return a1;
}

// Function: sub_2511E10
// Address: 0x2511e10
//
void __fastcall sub_2511E10(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // r9
  __m128i *v6; // r12
  unsigned __int64 v7; // rbx
  __m128i *v8; // r14
  __int8 *v9; // r12
  __int64 v10; // r15
  unsigned __int64 i; // rbx
  void (__fastcall *v12)(__int8 *, unsigned __int64, __int64); // rax
  __m128i *v13; // rax
  __m128i *v14; // rbx
  void (__fastcall *v15)(__m128i *, __m128i *, __int64); // rax
  __m128i *v16; // rbx
  void (__fastcall *v17)(__m128i *, __m128i *, __int64); // rax
  __int64 v18; // rbx
  void (__fastcall *v19)(__m128i *, __int64, __int64); // rax
  __m128i v20; // xmm0
  void (__fastcall *v21)(_QWORD, _QWORD, _QWORD); // rdx
  void (__fastcall *v22)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // r12
  __m128i *v26; // rcx
  void (__fastcall *v27)(__m128i *, __int64, __int64); // rax
  __m128i v28; // xmm0
  void (__fastcall *v29)(_QWORD, _QWORD, _QWORD); // rdx
  void (__fastcall *v30)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // [rsp-70h] [rbp-70h]
  unsigned __int64 v34; // [rsp-70h] [rbp-70h]
  unsigned int v35; // [rsp-64h] [rbp-64h]
  unsigned __int64 v36; // [rsp-60h] [rbp-60h]
  __int64 v37; // [rsp-60h] [rbp-60h]
  __m128i *v38; // [rsp-60h] [rbp-60h]
  __m128i *v39; // [rsp-60h] [rbp-60h]
  __m128i v40; // [rsp-58h] [rbp-58h] BYREF
  void (__fastcall *v41)(_QWORD, _QWORD, _QWORD); // [rsp-48h] [rbp-48h]
  __int64 v42; // [rsp-40h] [rbp-40h]

  if ( (__int64 *)a1 != a2 )
  {
    v6 = *(__m128i **)a1;
    v7 = *(unsigned int *)(a1 + 8);
    v35 = *((_DWORD *)a2 + 2);
    v5 = v35;
    v8 = *(__m128i **)a1;
    if ( v35 <= v7 )
    {
      v13 = *(__m128i **)a1;
      if ( v35 )
      {
        v18 = *a2;
        v33 = 2LL * v35;
        v37 = *a2 + v33 * 16;
        do
        {
          v41 = 0;
          v19 = *(void (__fastcall **)(__m128i *, __int64, __int64))(v18 + 16);
          if ( v19 )
          {
            v19(&v40, v18, 2);
            v42 = *(_QWORD *)(v18 + 24);
            v41 = *(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(v18 + 16);
          }
          v20 = _mm_loadu_si128(&v40);
          v40 = _mm_loadu_si128(v8);
          v21 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v8[1].m128i_i64[0];
          *v8 = v20;
          v22 = v41;
          v41 = v21;
          v23 = v8[1].m128i_i64[1];
          v8[1].m128i_i64[0] = (__int64)v22;
          v24 = v42;
          v42 = v23;
          v8[1].m128i_i64[1] = v24;
          if ( v41 )
            v41(&v40, &v40, 3);
          v18 += 32;
          v8 += 2;
        }
        while ( v18 != v37 );
        v13 = *(__m128i **)a1;
        v7 = *(unsigned int *)(a1 + 8);
        v8 = &v6[v33];
      }
      v14 = &v13[2 * v7];
      while ( v8 != v14 )
      {
        v15 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v14[-1].m128i_i64[0];
        v14 -= 2;
        if ( v15 )
          v15(v14, v14, 3);
      }
    }
    else
    {
      if ( v35 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v16 = &v6[2 * v7];
        while ( v6 != v16 )
        {
          while ( 1 )
          {
            v17 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v16[-1].m128i_i64[0];
            v16 -= 2;
            if ( !v17 )
              break;
            v36 = v5;
            v17(v16, v16, 3);
            v5 = v36;
            if ( v6 == v16 )
              goto LABEL_21;
          }
        }
LABEL_21:
        *(_DWORD *)(a1 + 8) = 0;
        v7 = 0;
        sub_2511D10(a1, v5, a3, a4, a5, v5);
        v5 = *((unsigned int *)a2 + 2);
        v6 = *(__m128i **)a1;
      }
      else if ( *(_DWORD *)(a1 + 8) )
      {
        v25 = *a2;
        v7 *= 32LL;
        v26 = &v40;
        v34 = *a2 + v7;
        do
        {
          v41 = 0;
          v27 = *(void (__fastcall **)(__m128i *, __int64, __int64))(v25 + 16);
          if ( v27 )
          {
            v38 = v26;
            v27(v26, v25, 2);
            v26 = v38;
            v42 = *(_QWORD *)(v25 + 24);
            v41 = *(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(v25 + 16);
          }
          v28 = _mm_loadu_si128(&v40);
          v40 = _mm_loadu_si128(v8);
          v29 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v8[1].m128i_i64[0];
          *v8 = v28;
          v30 = v41;
          v41 = v29;
          v31 = v8[1].m128i_i64[1];
          v8[1].m128i_i64[0] = (__int64)v30;
          v32 = v42;
          v42 = v31;
          v8[1].m128i_i64[1] = v32;
          if ( v41 )
          {
            v39 = v26;
            v41(v26, v26, 3);
            v26 = v39;
          }
          v25 += 32;
          v8 += 2;
        }
        while ( v25 != v34 );
        v5 = *((unsigned int *)a2 + 2);
        v6 = *(__m128i **)a1;
      }
      v9 = &v6->m128i_i8[v7];
      v10 = *a2 + 32 * v5;
      for ( i = *a2 + v7; v10 != i; v9 += 32 )
      {
        if ( v9 )
        {
          *((_QWORD *)v9 + 2) = 0;
          v12 = *(void (__fastcall **)(__int8 *, unsigned __int64, __int64))(i + 16);
          if ( v12 )
          {
            v12(v9, i, 2);
            *((_QWORD *)v9 + 3) = *(_QWORD *)(i + 24);
            *((_QWORD *)v9 + 2) = *(_QWORD *)(i + 16);
          }
        }
        i += 32LL;
      }
    }
    *(_DWORD *)(a1 + 8) = v35;
  }
}

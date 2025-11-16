// Function: sub_19D5E90
// Address: 0x19d5e90
//
__int64 __fastcall sub_19D5E90(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __m128i *a5, __m128i *a6, __m128i *a7)
{
  __int64 v8; // rax
  __m128i v9; // xmm1
  __int64 v10; // rcx
  __int64 v11; // rdx
  __m128i v12; // xmm0
  __m128i v13; // xmm2
  void (__fastcall *v14)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v15; // rcx
  __int64 v16; // rcx
  __int64 v17; // rax
  __m128i v18; // xmm3
  __m128i v19; // xmm0
  __int64 v20; // rdx
  __m128i v21; // xmm4
  void (__fastcall *v22)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v23; // rcx
  __int64 v24; // rcx
  __int64 v25; // rax
  __m128i v26; // xmm5
  __m128i v27; // xmm0
  __int64 v28; // rdx
  __m128i v29; // xmm6
  void (__fastcall *v30)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v31; // rcx
  unsigned int v32; // r12d
  __int64 v33; // rax
  unsigned int v34; // eax
  __m128i v36; // [rsp+0h] [rbp-40h] BYREF
  void (__fastcall *v37)(_QWORD, _QWORD, _QWORD); // [rsp+10h] [rbp-30h]
  __int64 v38; // [rsp+18h] [rbp-28h]

  v8 = v38;
  v9 = _mm_loadu_si128(&v36);
  a1->m128i_i64[0] = a3;
  a1->m128i_i64[1] = a4;
  v10 = a5[1].m128i_i64[0];
  v11 = a5[1].m128i_i64[1];
  v12 = _mm_loadu_si128(a5);
  a5[1].m128i_i64[0] = 0;
  a5[1].m128i_i64[1] = v8;
  *a5 = v9;
  v13 = _mm_loadu_si128(a1 + 1);
  v14 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))a1[2].m128i_i64[0];
  a1[2].m128i_i64[0] = v10;
  v15 = a1[2].m128i_i64[1];
  v37 = v14;
  v38 = v15;
  a1[2].m128i_i64[1] = v11;
  v36 = v13;
  a1[1] = v12;
  if ( v14 )
    v14(&v36, &v36, 3);
  v16 = a6[1].m128i_i64[0];
  v17 = v38;
  a6[1].m128i_i64[0] = 0;
  v18 = _mm_loadu_si128(&v36);
  v19 = _mm_loadu_si128(a6);
  v20 = a6[1].m128i_i64[1];
  a6[1].m128i_i64[1] = v17;
  *a6 = v18;
  v21 = _mm_loadu_si128(a1 + 3);
  v22 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))a1[4].m128i_i64[0];
  a1[4].m128i_i64[0] = v16;
  v23 = a1[4].m128i_i64[1];
  v37 = v22;
  v38 = v23;
  a1[4].m128i_i64[1] = v20;
  v36 = v21;
  a1[3] = v19;
  if ( v22 )
    v22(&v36, &v36, 3);
  v24 = a7[1].m128i_i64[0];
  v25 = v38;
  a7[1].m128i_i64[0] = 0;
  v26 = _mm_loadu_si128(&v36);
  v27 = _mm_loadu_si128(a7);
  v28 = a7[1].m128i_i64[1];
  a7[1].m128i_i64[1] = v25;
  *a7 = v26;
  v29 = _mm_loadu_si128(a1 + 5);
  v30 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))a1[6].m128i_i64[0];
  a1[6].m128i_i64[0] = v24;
  v31 = a1[6].m128i_i64[1];
  v37 = v30;
  v38 = v31;
  a1[6].m128i_i64[1] = v28;
  v36 = v29;
  a1[5] = v27;
  if ( v30 )
    v30(&v36, &v36, 3);
  v32 = 0;
  v33 = *(_QWORD *)a1->m128i_i64[1];
  if ( (*(_BYTE *)(v33 + 73) & 0x30) != 0 && (*(_BYTE *)(v33 + 72) & 0x30) != 0 )
  {
    v34 = 0;
    do
    {
      v32 = v34;
      v34 = sub_19D5B50(
              (__int64)a1,
              a2,
              v28,
              *(double *)v27.m128i_i64,
              *(double *)v9.m128i_i64,
              *(double *)v13.m128i_i64);
    }
    while ( (_BYTE)v34 );
    a1->m128i_i64[0] = 0;
  }
  return v32;
}

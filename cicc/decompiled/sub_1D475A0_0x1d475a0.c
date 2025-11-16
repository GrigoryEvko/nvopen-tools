// Function: sub_1D475A0
// Address: 0x1d475a0
//
__int64 __fastcall sub_1D475A0(__int64 *a1)
{
  __int64 (__fastcall *v1)(__int64); // rax
  _QWORD *v2; // rax
  __m128i v3; // xmm0
  __int64 v4; // rax
  __int64 (__fastcall *v5)(__int64); // rax
  _QWORD *v6; // rax
  __m128i v7; // xmm1
  __int64 v8; // rax
  __int64 (__fastcall *v9)(__int64); // rax
  _QWORD *v10; // rax
  __m128i v11; // xmm2
  __m128i v13; // [rsp+10h] [rbp-30h] BYREF
  int v14; // [rsp+20h] [rbp-20h]

  v1 = *(__int64 (__fastcall **)(__int64))(*a1 + 160);
  if ( v1 == sub_1D47530 )
  {
    v13 = 0u;
    v14 = 8;
    v2 = (_QWORD *)malloc(8u);
    if ( !v2 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v2 = 0;
    }
    v13.m128i_i64[0] = (__int64)v2;
    v13.m128i_i64[1] = 1;
    *v2 = 0;
  }
  else
  {
    ((void (__fastcall *)(__m128i *, __int64 *))v1)(&v13, a1);
  }
  _libc_free(a1[20]);
  v3 = _mm_loadu_si128(&v13);
  *((_DWORD *)a1 + 44) = v14;
  v4 = *a1;
  *((__m128i *)a1 + 10) = v3;
  v5 = *(__int64 (__fastcall **)(__int64))(v4 + 168);
  if ( v5 == sub_1D474C0 )
  {
    v13 = 0u;
    v14 = 8;
    v6 = (_QWORD *)malloc(8u);
    if ( !v6 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v6 = 0;
    }
    v13.m128i_i64[0] = (__int64)v6;
    v13.m128i_i64[1] = 1;
    *v6 = 0;
  }
  else
  {
    ((void (__fastcall *)(__m128i *, __int64 *))v5)(&v13, a1);
  }
  _libc_free(a1[23]);
  v7 = _mm_loadu_si128(&v13);
  *((_DWORD *)a1 + 50) = v14;
  v8 = *a1;
  *(__m128i *)(a1 + 23) = v7;
  v9 = *(__int64 (__fastcall **)(__int64))(v8 + 176);
  if ( v9 == sub_1D47450 )
  {
    v13 = 0u;
    v14 = 8;
    v10 = (_QWORD *)malloc(8u);
    if ( !v10 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v10 = 0;
    }
    v13.m128i_i64[0] = (__int64)v10;
    v13.m128i_i64[1] = 1;
    *v10 = 0;
  }
  else
  {
    ((void (__fastcall *)(__m128i *, __int64 *))v9)(&v13, a1);
  }
  _libc_free(a1[26]);
  v11 = _mm_loadu_si128(&v13);
  *((_DWORD *)a1 + 56) = v14;
  *((__m128i *)a1 + 13) = v11;
  return 0;
}

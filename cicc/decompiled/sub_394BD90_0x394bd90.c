// Function: sub_394BD90
// Address: 0x394bd90
//
unsigned __int64 __fastcall sub_394BD90(
        unsigned __int64 *a1,
        unsigned int a2,
        __m128i *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 result; // rax
  __m128i v7; // xmm0
  __int64 (__fastcall *v8)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _DWORD); // rcx
  __m128i v9; // xmm1
  __int64 v10; // r8
  __int64 v11; // rsi
  unsigned __int64 v12; // rsi
  __m128i v13; // xmm2
  unsigned __int64 v14; // rdx
  __m128i v15; // xmm0
  __int64 (__fastcall *v16)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _DWORD); // rax
  unsigned int v17; // [rsp+0h] [rbp-30h] BYREF
  __m128i v18; // [rsp+8h] [rbp-28h] BYREF
  __int64 (__fastcall *v19)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _DWORD); // [rsp+18h] [rbp-18h]
  unsigned __int64 v20; // [rsp+20h] [rbp-10h]

  result = a2;
  v7 = _mm_loadu_si128(a3);
  v8 = (__int64 (__fastcall *)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _DWORD))a3[1].m128i_i64[0];
  v9 = _mm_loadu_si128(&v18);
  v10 = v20;
  v17 = a2;
  v11 = a3[1].m128i_i64[1];
  a3[1].m128i_i64[0] = 0;
  a3[1].m128i_i64[1] = v10;
  v20 = v11;
  *a3 = v9;
  v12 = a1[20];
  v19 = v8;
  v18 = v7;
  if ( v12 == a1[21] )
  {
    result = sub_394BB20(a1 + 19, (__m128i *)v12, (__int64)&v17);
    v8 = v19;
  }
  else
  {
    if ( v12 )
    {
      v13 = _mm_loadu_si128((const __m128i *)(v12 + 8));
      v14 = *(_QWORD *)(v12 + 32);
      *(_DWORD *)v12 = result;
      v15 = _mm_loadu_si128(&v18);
      *(_QWORD *)(v12 + 24) = 0;
      v18 = v13;
      *(__m128i *)(v12 + 8) = v15;
      v16 = v19;
      v19 = 0;
      *(_QWORD *)(v12 + 24) = v16;
      result = v20;
      v8 = v19;
      v20 = v14;
      *(_QWORD *)(v12 + 32) = result;
      v12 = a1[20];
    }
    a1[20] = v12 + 40;
  }
  if ( v8 )
    return v8(&v18, &v18, 3, v8, v10, a6, v17);
  return result;
}

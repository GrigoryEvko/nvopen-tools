// Function: sub_25BD6E0
// Address: 0x25bd6e0
//
__int64 __fastcall sub_25BD6E0(__int64 a1, __m128i *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __m128i v5; // xmm0
  __int64 v6; // rcx
  __int64 v7; // rdi
  __int64 v8; // rcx
  __m128i v9; // xmm2
  __m128i v10; // xmm0
  __int64 v11; // rcx
  __int64 v12; // rdi
  __int64 v13; // rcx
  __m128i v14; // xmm3
  __m128i v15; // xmm0
  __int64 v16; // rcx
  __int64 v17; // rcx
  __int64 v18; // r12
  __int64 v19; // rbx
  void (__fastcall *v20)(__int64, __int64, __int64); // rax
  void (__fastcall *v21)(__int64, __int64, __int64); // rax

  result = *(_QWORD *)a1;
  v3 = *(_QWORD *)a1 + 104LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v3 )
  {
    do
    {
      if ( a2 )
      {
        a2[1].m128i_i64[0] = 0;
        v5 = _mm_loadu_si128((const __m128i *)result);
        *(__m128i *)result = _mm_loadu_si128(a2);
        *a2 = v5;
        v6 = *(_QWORD *)(result + 16);
        *(_QWORD *)(result + 16) = 0;
        v7 = a2[1].m128i_i64[1];
        a2[1].m128i_i64[0] = v6;
        v8 = *(_QWORD *)(result + 24);
        *(_QWORD *)(result + 24) = v7;
        v9 = _mm_loadu_si128(a2 + 2);
        a2[3].m128i_i64[0] = 0;
        a2[1].m128i_i64[1] = v8;
        v10 = _mm_loadu_si128((const __m128i *)(result + 32));
        *(__m128i *)(result + 32) = v9;
        a2[2] = v10;
        v11 = *(_QWORD *)(result + 48);
        *(_QWORD *)(result + 48) = 0;
        v12 = a2[3].m128i_i64[1];
        a2[3].m128i_i64[0] = v11;
        v13 = *(_QWORD *)(result + 56);
        *(_QWORD *)(result + 56) = v12;
        v14 = _mm_loadu_si128(a2 + 4);
        a2[5].m128i_i64[0] = 0;
        a2[3].m128i_i64[1] = v13;
        v15 = _mm_loadu_si128((const __m128i *)(result + 64));
        *(__m128i *)(result + 64) = v14;
        a2[4] = v15;
        v16 = *(_QWORD *)(result + 80);
        *(_QWORD *)(result + 80) = 0;
        a2[5].m128i_i64[0] = v16;
        v17 = *(_QWORD *)(result + 88);
        *(_QWORD *)(result + 88) = a2[5].m128i_i64[1];
        a2[5].m128i_i64[1] = v17;
        a2[6].m128i_i32[0] = *(_DWORD *)(result + 96);
        a2[6].m128i_i8[4] = *(_BYTE *)(result + 100);
      }
      result += 104;
      a2 = (__m128i *)((char *)a2 + 104);
    }
    while ( v3 != result );
    v18 = *(_QWORD *)a1;
    result = 13LL * *(unsigned int *)(a1 + 8);
    v19 = *(_QWORD *)a1 + 104LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v19 )
    {
      do
      {
        v20 = *(void (__fastcall **)(__int64, __int64, __int64))(v19 - 24);
        v19 -= 104;
        if ( v20 )
          v20(v19 + 64, v19 + 64, 3);
        v21 = *(void (__fastcall **)(__int64, __int64, __int64))(v19 + 48);
        if ( v21 )
          v21(v19 + 32, v19 + 32, 3);
        result = *(_QWORD *)(v19 + 16);
        if ( result )
          result = ((__int64 (__fastcall *)(__int64, __int64, __int64))result)(v19, v19, 3);
      }
      while ( v19 != v18 );
    }
  }
  return result;
}

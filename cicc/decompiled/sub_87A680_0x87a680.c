// Function: sub_87A680
// Address: 0x87a680
//
__int64 __fastcall sub_87A680(__m128i *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rcx
  __int64 result; // rax
  __int64 *v5; // rax
  __int64 v6; // rdx

  v3 = a1[1].m128i_i64[1];
  if ( *(_BYTE *)(v3 + 80) == 19 )
    v3 = *(_QWORD *)(*(_QWORD *)(v3 + 88) + 176LL);
  result = qword_4F600E0;
  if ( a1->m128i_i64[0] != qword_4F600E0 )
  {
    if ( (_DWORD)a3 || (v5 = *(__int64 **)(*(_QWORD *)(v3 + 96) + 8LL)) == 0 )
    {
      result = sub_877070(a1, a2, a3, v3);
      *(_QWORD *)(result + 8) = *(_QWORD *)(a1->m128i_i64[0] + 8);
      *(_QWORD *)(result + 16) = *(_QWORD *)(a1->m128i_i64[0] + 16);
    }
    else
    {
      result = *v5;
    }
  }
  *a1 = _mm_loadu_si128(xmmword_4F06660);
  a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
  a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
  a1[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
  v6 = *a2;
  a1->m128i_i64[0] = result;
  a1->m128i_i64[1] = v6;
  return result;
}

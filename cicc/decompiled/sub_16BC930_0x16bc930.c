// Function: sub_16BC930
// Address: 0x16bc930
//
_BYTE *__fastcall sub_16BC930(__int64 a1, __int64 a2)
{
  __m128i *v2; // rdx
  _BYTE *result; // rax
  __m128i si128; // xmm0
  _QWORD *v5; // r12
  _QWORD *i; // r13

  v2 = *(__m128i **)(a2 + 24);
  result = (_BYTE *)(*(_QWORD *)(a2 + 16) - (_QWORD)v2);
  if ( (unsigned __int64)result <= 0x10 )
  {
    result = (_BYTE *)sub_16E7EE0(a2, "Multiple errors:\n", 17);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F668B0);
    v2[1].m128i_i8[0] = 10;
    *v2 = si128;
    *(_QWORD *)(a2 + 24) += 17LL;
  }
  v5 = *(_QWORD **)(a1 + 8);
  for ( i = *(_QWORD **)(a1 + 16); i != v5; result = (_BYTE *)sub_16E7EE0(a2, "\n", 1) )
  {
    while ( 1 )
    {
      (*(void (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v5 + 16LL))(*v5, a2);
      result = *(_BYTE **)(a2 + 24);
      if ( *(_BYTE **)(a2 + 16) == result )
        break;
      ++v5;
      *result = 10;
      ++*(_QWORD *)(a2 + 24);
      if ( i == v5 )
        return result;
    }
    ++v5;
  }
  return result;
}

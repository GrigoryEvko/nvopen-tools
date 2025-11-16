// Function: sub_2EAFF00
// Address: 0x2eaff00
//
_BYTE *__fastcall sub_2EAFF00(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __m128i *v6; // rdx
  __m128i si128; // xmm0
  _BYTE *result; // rax

  v6 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v6 <= 0x10u )
  {
    sub_CB6200(a2, "machine-function(", 0x11u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4450D60);
    v6[1].m128i_i8[0] = 40;
    *v6 = si128;
    *(_QWORD *)(a2 + 32) += 17LL;
  }
  (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64))(*(_QWORD *)*a1 + 24LL))(*a1, a2, a3, a4);
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 41);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 41;
  return result;
}

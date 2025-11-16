// Function: sub_24E4120
// Address: 0x24e4120
//
_BYTE *__fastcall sub_24E4120(__int64 a1, __int64 a2)
{
  __m128i *v2; // rdx
  __m128i si128; // xmm0
  _BYTE *result; // rax

  v2 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v2 <= 0x19u )
  {
    sub_CB6200(a2, "While splitting coroutine ", 0x1Au);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4388A50);
    qmemcpy(&v2[1], "coroutine ", 10);
    *v2 = si128;
    *(_QWORD *)(a2 + 32) += 26LL;
  }
  sub_A5BF40(*(unsigned __int8 **)(a1 + 16), a2, 0, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 40LL));
  result = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == result )
    return (_BYTE *)sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
  *result = 10;
  ++*(_QWORD *)(a2 + 32);
  return result;
}

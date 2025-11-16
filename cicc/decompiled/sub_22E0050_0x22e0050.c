// Function: sub_22E0050
// Address: 0x22e0050
//
unsigned __int64 __fastcall sub_22E0050(__int64 a1, __int64 a2)
{
  void *v2; // rdx
  __m128i *v3; // rdx
  unsigned __int64 result; // rax

  v2 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v2 <= 0xCu )
  {
    sub_CB6200(a2, "Region tree:\n", 0xDu);
  }
  else
  {
    qmemcpy(v2, "Region tree:\n", 13);
    *(_QWORD *)(a2 + 32) += 13LL;
  }
  sub_22DF440(*(__int64 **)(a1 + 32), a2, 1, 0, unk_4FDC048);
  v3 = *(__m128i **)(a2 + 32);
  result = *(_QWORD *)(a2 + 24) - (_QWORD)v3;
  if ( result <= 0xF )
    return sub_CB6200(a2, "End region tree\n", 0x10u);
  *v3 = _mm_load_si128((const __m128i *)&xmmword_428CBD0);
  *(_QWORD *)(a2 + 32) += 16LL;
  return result;
}

// Function: sub_14478D0
// Address: 0x14478d0
//
unsigned __int64 __fastcall sub_14478D0(__int64 a1, __int64 a2)
{
  void *v2; // rdx
  __m128i *v3; // rdx
  unsigned __int64 result; // rax

  v2 = *(void **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v2 <= 0xCu )
  {
    sub_16E7EE0(a2, "Region tree:\n", 13);
  }
  else
  {
    qmemcpy(v2, "Region tree:\n", 13);
    *(_QWORD *)(a2 + 24) += 13LL;
  }
  sub_1446CA0(*(__int64 **)(a1 + 32), a2, 1, 0, unk_4F9A388);
  v3 = *(__m128i **)(a2 + 24);
  result = *(_QWORD *)(a2 + 16) - (_QWORD)v3;
  if ( result <= 0xF )
    return sub_16E7EE0(a2, "End region tree\n", 16);
  *v3 = _mm_load_si128((const __m128i *)&xmmword_428CBD0);
  *(_QWORD *)(a2 + 24) += 16LL;
  return result;
}

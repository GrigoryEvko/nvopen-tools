// Function: sub_D9A160
// Address: 0xd9a160
//
__int64 __fastcall sub_D9A160(__int64 a1, int a2)
{
  __int64 v3; // rdx
  void *v4; // rdx
  __m128i *v5; // rdx
  __m128i si128; // xmm0

  switch ( a2 )
  {
    case 1:
      v3 = *(_QWORD *)(a1 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v3) <= 8 )
      {
        sub_CB6200(a1, (unsigned __int8 *)"Dominates", 9u);
      }
      else
      {
        *(_BYTE *)(v3 + 8) = 115;
        *(_QWORD *)v3 = 0x6574616E696D6F44LL;
        *(_QWORD *)(a1 + 32) += 9LL;
      }
      break;
    case 2:
      v5 = *(__m128i **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v5 <= 0x10u )
      {
        sub_CB6200(a1, "ProperlyDominates", 0x11u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F74EC0);
        v5[1].m128i_i8[0] = 115;
        *v5 = si128;
        *(_QWORD *)(a1 + 32) += 17LL;
      }
      break;
    case 0:
      v4 = *(void **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v4 <= 0xEu )
      {
        sub_CB6200(a1, "DoesNotDominate", 0xFu);
      }
      else
      {
        qmemcpy(v4, "DoesNotDominate", 15);
        *(_QWORD *)(a1 + 32) += 15LL;
      }
      break;
  }
  return a1;
}

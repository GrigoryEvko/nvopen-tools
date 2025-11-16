// Function: sub_FD6EF0
// Address: 0xfd6ef0
//
_BYTE *__fastcall sub_FD6EF0(__int64 a1, __int64 a2)
{
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // r13
  unsigned __int64 i; // rsi
  __m128i *v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rax
  __m128i *v13; // rdx
  __m128i v14; // xmm0
  unsigned __int64 j; // rbx
  _BYTE *result; // rax

  v4 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 <= 0x12u )
  {
    v6 = sub_CB6200(a2, "Alias Set Tracker: ", 0x13u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F8C2E0);
    v4[1].m128i_i8[2] = 32;
    v6 = a2;
    v4[1].m128i_i16[0] = 14962;
    *v4 = si128;
    *(_QWORD *)(a2 + 32) += 19LL;
  }
  v7 = *(_QWORD *)(a1 + 16);
  v8 = a1 + 8;
  for ( i = 0; v7 != v8; ++i )
    v7 = *(_QWORD *)(v7 + 8);
  sub_CB59D0(v6, i);
  v10 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a1 + 64) )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v10 <= 0xBu )
    {
      sub_CB6200(a2, " (Saturated)", 0xCu);
      v10 = *(__m128i **)(a2 + 32);
    }
    else
    {
      qmemcpy(v10, " (Saturated)", 12);
      v10 = (__m128i *)(*(_QWORD *)(a2 + 32) + 12LL);
      *(_QWORD *)(a2 + 32) = v10;
    }
  }
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v10 <= 0xFu )
  {
    v11 = sub_CB6200(a2, " alias sets for ", 0x10u);
  }
  else
  {
    v11 = a2;
    *v10 = _mm_load_si128((const __m128i *)&xmmword_3F8C2F0);
    *(_QWORD *)(a2 + 32) += 16LL;
  }
  v12 = sub_CB59D0(v11, *(unsigned int *)(a1 + 40));
  v13 = *(__m128i **)(v12 + 32);
  if ( *(_QWORD *)(v12 + 24) - (_QWORD)v13 <= 0x10u )
  {
    sub_CB6200(v12, " pointer values.\n", 0x11u);
  }
  else
  {
    v14 = _mm_load_si128((const __m128i *)&xmmword_3F8C300);
    v13[1].m128i_i8[0] = 10;
    *v13 = v14;
    *(_QWORD *)(v12 + 32) += 17LL;
  }
  for ( j = *(_QWORD *)(a1 + 16); v8 != j; j = *(_QWORD *)(j + 8) )
    sub_FD66F0(j, a2);
  result = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == result )
    return (_BYTE *)sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
  *result = 10;
  ++*(_QWORD *)(a2 + 32);
  return result;
}

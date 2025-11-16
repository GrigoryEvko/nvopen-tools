// Function: sub_1359660
// Address: 0x1359660
//
_BYTE *__fastcall sub_1359660(__int64 a1, __int64 a2)
{
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 i; // rsi
  __int64 v10; // rax
  __m128i *v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rax
  __m128i *v14; // rdx
  __m128i v15; // xmm0
  __int64 j; // rbx
  _BYTE *result; // rax

  v4 = *(__m128i **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v4 <= 0x12u )
  {
    v6 = sub_16E7EE0(a2, "Alias Set Tracker: ", 19);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F8C2E0);
    v4[1].m128i_i8[2] = 32;
    v6 = a2;
    v4[1].m128i_i16[0] = 14962;
    *v4 = si128;
    *(_QWORD *)(a2 + 24) += 19LL;
  }
  v7 = *(_QWORD *)(a1 + 16);
  v8 = a1 + 8;
  for ( i = 0; v8 != v7; ++i )
    v7 = *(_QWORD *)(v7 + 8);
  v10 = sub_16E7A90(v6, i);
  v11 = *(__m128i **)(v10 + 24);
  v12 = v10;
  if ( *(_QWORD *)(v10 + 16) - (_QWORD)v11 <= 0xFu )
  {
    v12 = sub_16E7EE0(v10, " alias sets for ", 16);
  }
  else
  {
    *v11 = _mm_load_si128((const __m128i *)&xmmword_3F8C2F0);
    *(_QWORD *)(v10 + 24) += 16LL;
  }
  v13 = sub_16E7A90(v12, *(unsigned int *)(a1 + 40));
  v14 = *(__m128i **)(v13 + 24);
  if ( *(_QWORD *)(v13 + 16) - (_QWORD)v14 <= 0x10u )
  {
    sub_16E7EE0(v13, " pointer values.\n", 17);
  }
  else
  {
    v15 = _mm_load_si128((const __m128i *)&xmmword_3F8C300);
    v14[1].m128i_i8[0] = 10;
    *v14 = v15;
    *(_QWORD *)(v13 + 24) += 17LL;
  }
  for ( j = *(_QWORD *)(a1 + 16); v8 != j; j = *(_QWORD *)(j + 8) )
    sub_1358EC0(j, a2);
  result = *(_BYTE **)(a2 + 24);
  if ( *(_BYTE **)(a2 + 16) == result )
    return (_BYTE *)sub_16E7EE0(a2, "\n", 1);
  *result = 10;
  ++*(_QWORD *)(a2 + 24);
  return result;
}

// Function: sub_1456A80
// Address: 0x1456a80
//
_BYTE *__fastcall sub_1456A80(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // rax
  __m128i *v4; // rdx
  __int64 v5; // r12
  __m128i si128; // xmm0
  _DWORD *v7; // rdx
  _BYTE *result; // rax

  v3 = sub_16E8750(a2, a3);
  v4 = *(__m128i **)(v3 + 24);
  v5 = v3;
  if ( *(_QWORD *)(v3 + 16) - (_QWORD)v4 <= 0x10u )
  {
    v5 = sub_16E7EE0(v3, "Equal predicate: ", 17);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F74EA0);
    v4[1].m128i_i8[0] = 32;
    *v4 = si128;
    *(_QWORD *)(v3 + 24) += 17LL;
  }
  sub_1456620(*(_QWORD *)(a1 + 40), v5);
  v7 = *(_DWORD **)(v5 + 24);
  if ( *(_QWORD *)(v5 + 16) - (_QWORD)v7 <= 3u )
  {
    v5 = sub_16E7EE0(v5, " == ", 4);
  }
  else
  {
    *v7 = 540884256;
    *(_QWORD *)(v5 + 24) += 4LL;
  }
  sub_1456620(*(_QWORD *)(a1 + 48), v5);
  result = *(_BYTE **)(v5 + 24);
  if ( *(_BYTE **)(v5 + 16) == result )
    return (_BYTE *)sub_16E7EE0(v5, "\n", 1);
  *result = 10;
  ++*(_QWORD *)(v5 + 24);
  return result;
}

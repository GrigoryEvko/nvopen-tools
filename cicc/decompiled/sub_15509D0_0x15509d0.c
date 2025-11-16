// Function: sub_15509D0
// Address: 0x15509d0
//
_BYTE *__fastcall sub_15509D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __m128i *v8; // rdx
  __m128i si128; // xmm0
  __int64 v10; // rcx
  _BYTE *result; // rax
  __int64 v12; // [rsp+0h] [rbp-60h] BYREF
  char v13; // [rsp+8h] [rbp-58h]
  char *v14; // [rsp+10h] [rbp-50h]
  __int64 v15; // [rsp+18h] [rbp-48h]
  __int64 v16; // [rsp+20h] [rbp-40h]
  __int64 v17; // [rsp+28h] [rbp-38h]

  v8 = *(__m128i **)(a1 + 24);
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v8 <= 0x12u )
  {
    sub_16E7EE0(a1, "!DIFortranSubrange(", 19);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4293140);
    v8[1].m128i_i8[2] = 40;
    v8[1].m128i_i16[0] = 25959;
    *v8 = si128;
    *(_QWORD *)(a1 + 24) += 19LL;
  }
  v17 = a5;
  v10 = *(_QWORD *)(a2 + 24);
  v12 = a1;
  v13 = 1;
  v14 = ", ";
  v15 = a3;
  v16 = a4;
  sub_154AEF0((__int64)&v12, "constLowerBound", 0xFu, v10, 0);
  if ( !*(_BYTE *)(a2 + 40) )
    sub_154AEF0((__int64)&v12, "constUpperBound", 0xFu, *(_QWORD *)(a2 + 32), 0);
  sub_154F950((__int64)&v12, "lowerBound", 0xAu, *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8)), 1);
  sub_154F950(
    (__int64)&v12,
    "lowerBoundExpression",
    0x14u,
    *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8))),
    1);
  sub_154F950((__int64)&v12, "upperBound", 0xAu, *(unsigned __int8 **)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8))), 1);
  sub_154F950(
    (__int64)&v12,
    "upperBoundExpression",
    0x14u,
    *(unsigned __int8 **)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8))),
    1);
  result = *(_BYTE **)(a1 + 24);
  if ( *(_BYTE **)(a1 + 16) == result )
    return (_BYTE *)sub_16E7EE0(a1, ")", 1);
  *result = 41;
  ++*(_QWORD *)(a1 + 24);
  return result;
}

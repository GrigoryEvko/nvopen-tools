// Function: sub_CA17B0
// Address: 0xca17b0
//
_BYTE *__fastcall sub_CA17B0(const char *src)
{
  __int64 v2; // rax
  __m128i *v3; // rdx
  __int64 v4; // r12
  __m128i si128; // xmm0
  size_t v6; // rax
  _BYTE *v7; // rdi
  size_t v8; // r14
  _BYTE *result; // rax

  if ( !qword_4F84F80 )
    sub_C7D570(&qword_4F84F80, sub_CA1590, (__int64)sub_CA1570);
  if ( !*(_BYTE *)(qword_4F84F80 + 136) )
    sub_C64ED0("Invalid size request on a scalable vector.", 1u);
  v2 = sub_CA5BD0();
  v3 = *(__m128i **)(v2 + 32);
  v4 = v2;
  if ( *(_QWORD *)(v2 + 24) - (_QWORD)v3 <= 0x2Au )
  {
    v4 = sub_CB6200(v2, "Invalid size request on a scalable vector; ", 43);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F67C30);
    qmemcpy(&v3[2], "le vector; ", 11);
    *v3 = si128;
    v3[1] = _mm_load_si128((const __m128i *)&xmmword_3F67C40);
    *(_QWORD *)(v2 + 32) += 43LL;
  }
  if ( !src )
    goto LABEL_12;
  v6 = strlen(src);
  v7 = *(_BYTE **)(v4 + 32);
  v8 = v6;
  result = *(_BYTE **)(v4 + 24);
  if ( v8 > result - v7 )
  {
    v4 = sub_CB6200(v4, src, v8);
LABEL_12:
    result = *(_BYTE **)(v4 + 24);
    v7 = *(_BYTE **)(v4 + 32);
LABEL_13:
    if ( v7 == result )
      return (_BYTE *)sub_CB6200(v4, "\n", 1);
    goto LABEL_14;
  }
  if ( !v8 )
    goto LABEL_13;
  memcpy(v7, src, v8);
  result = *(_BYTE **)(v4 + 24);
  v7 = (_BYTE *)(v8 + *(_QWORD *)(v4 + 32));
  *(_QWORD *)(v4 + 32) = v7;
  if ( v7 == result )
    return (_BYTE *)sub_CB6200(v4, "\n", 1);
LABEL_14:
  *v7 = 10;
  ++*(_QWORD *)(v4 + 32);
  return result;
}

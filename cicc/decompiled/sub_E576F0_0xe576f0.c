// Function: sub_E576F0
// Address: 0xe576f0
//
_BYTE *__fastcall sub_E576F0(__int64 a1, unsigned int a2)
{
  __int64 v3; // rdi
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  unsigned __int64 v6; // r13
  bool v7; // zf
  _BYTE *result; // rax
  __int64 v9; // rdi
  __int64 v10; // r14
  unsigned __int8 *v11; // rsi
  size_t v12; // rdx
  void *v13; // rdi

  sub_E9C120();
  v3 = *(_QWORD *)(a1 + 304);
  v4 = *(__m128i **)(v3 + 32);
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v4 <= 0x10u )
  {
    v3 = sub_CB6200(v3, "\t.seh_stackalloc ", 0x11u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7F900);
    v4[1].m128i_i8[0] = 32;
    *v4 = si128;
    *(_QWORD *)(v3 + 32) += 17LL;
  }
  sub_CB59D0(v3, a2);
  v6 = *(_QWORD *)(a1 + 344);
  if ( v6 )
  {
    v10 = *(_QWORD *)(a1 + 304);
    v11 = *(unsigned __int8 **)(a1 + 336);
    v12 = *(_QWORD *)(a1 + 344);
    v13 = *(void **)(v10 + 32);
    if ( v6 > *(_QWORD *)(v10 + 24) - (_QWORD)v13 )
    {
      sub_CB6200(*(_QWORD *)(a1 + 304), v11, v12);
    }
    else
    {
      memcpy(v13, v11, v12);
      *(_QWORD *)(v10 + 32) += v6;
    }
  }
  v7 = *(_BYTE *)(a1 + 745) == 0;
  *(_QWORD *)(a1 + 344) = 0;
  if ( !v7 )
    return (_BYTE *)sub_E4D630((__int64 *)a1);
  v9 = *(_QWORD *)(a1 + 304);
  result = *(_BYTE **)(v9 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v9 + 24) )
    return (_BYTE *)sub_CB5D20(v9, 10);
  *(_QWORD *)(v9 + 32) = result + 1;
  *result = 10;
  return result;
}

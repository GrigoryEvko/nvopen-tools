// Function: sub_E57290
// Address: 0xe57290
//
_BYTE *__fastcall sub_E57290(__int64 a1)
{
  __int64 v2; // rdi
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  unsigned __int64 v5; // r13
  bool v6; // zf
  _BYTE *result; // rax
  __int64 v8; // rdi
  __int64 v9; // r14
  unsigned __int8 *v10; // rsi
  size_t v11; // rdx
  void *v12; // rdi

  sub_E99830();
  v2 = *(_QWORD *)(a1 + 304);
  v3 = *(__m128i **)(v2 + 32);
  if ( *(_QWORD *)(v2 + 24) - (_QWORD)v3 <= 0x10u )
  {
    sub_CB6200(v2, "\t.seh_endprologue", 0x11u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7F8F0);
    v3[1].m128i_i8[0] = 101;
    *v3 = si128;
    *(_QWORD *)(v2 + 32) += 17LL;
  }
  v5 = *(_QWORD *)(a1 + 344);
  if ( v5 )
  {
    v9 = *(_QWORD *)(a1 + 304);
    v10 = *(unsigned __int8 **)(a1 + 336);
    v11 = *(_QWORD *)(a1 + 344);
    v12 = *(void **)(v9 + 32);
    if ( v5 > *(_QWORD *)(v9 + 24) - (_QWORD)v12 )
    {
      sub_CB6200(*(_QWORD *)(a1 + 304), v10, v11);
    }
    else
    {
      memcpy(v12, v10, v11);
      *(_QWORD *)(v9 + 32) += v5;
    }
  }
  v6 = *(_BYTE *)(a1 + 745) == 0;
  *(_QWORD *)(a1 + 344) = 0;
  if ( !v6 )
    return (_BYTE *)sub_E4D630((__int64 *)a1);
  v8 = *(_QWORD *)(a1 + 304);
  result = *(_BYTE **)(v8 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v8 + 24) )
    return (_BYTE *)sub_CB5D20(v8, 10);
  *(_QWORD *)(v8 + 32) = result + 1;
  *result = 10;
  return result;
}

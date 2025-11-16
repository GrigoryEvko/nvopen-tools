// Function: sub_E57CE0
// Address: 0xe57ce0
//
_BYTE *__fastcall sub_E57CE0(__int64 a1)
{
  __int64 v2; // rdi
  __m128i *v3; // rdx
  unsigned __int64 v4; // r13
  bool v5; // zf
  _BYTE *result; // rax
  __int64 v7; // rdi
  __int64 v8; // r14
  unsigned __int8 *v9; // rsi
  size_t v10; // rdx
  void *v11; // rdi

  sub_E99600();
  v2 = *(_QWORD *)(a1 + 304);
  v3 = *(__m128i **)(v2 + 32);
  if ( *(_QWORD *)(v2 + 24) - (_QWORD)v3 <= 0xFu )
  {
    sub_CB6200(v2, "\t.seh_endfunclet", 0x10u);
  }
  else
  {
    *v3 = _mm_load_si128((const __m128i *)&xmmword_3F7F930);
    *(_QWORD *)(v2 + 32) += 16LL;
  }
  v4 = *(_QWORD *)(a1 + 344);
  if ( v4 )
  {
    v8 = *(_QWORD *)(a1 + 304);
    v9 = *(unsigned __int8 **)(a1 + 336);
    v10 = *(_QWORD *)(a1 + 344);
    v11 = *(void **)(v8 + 32);
    if ( v4 > *(_QWORD *)(v8 + 24) - (_QWORD)v11 )
    {
      sub_CB6200(*(_QWORD *)(a1 + 304), v9, v10);
    }
    else
    {
      memcpy(v11, v9, v10);
      *(_QWORD *)(v8 + 32) += v4;
    }
  }
  v5 = *(_BYTE *)(a1 + 745) == 0;
  *(_QWORD *)(a1 + 344) = 0;
  if ( !v5 )
    return (_BYTE *)sub_E4D630((__int64 *)a1);
  v7 = *(_QWORD *)(a1 + 304);
  result = *(_BYTE **)(v7 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v7 + 24) )
    return (_BYTE *)sub_CB5D20(v7, 10);
  *(_QWORD *)(v7 + 32) = result + 1;
  *result = 10;
  return result;
}

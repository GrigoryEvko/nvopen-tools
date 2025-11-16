// Function: sub_E50CE0
// Address: 0xe50ce0
//
_BYTE *__fastcall sub_E50CE0(__int64 a1, __int64 *a2, __int64 a3, int a4)
{
  __int64 v5; // rdi
  __m128i *v6; // rdx
  __m128i si128; // xmm0
  unsigned __int64 v8; // r13
  bool v9; // zf
  _BYTE *result; // rax
  __int64 v11; // rdi
  __int64 v12; // r14
  unsigned __int8 *v13; // rsi
  size_t v14; // rdx
  void *v15; // rdi

  sub_E4CF20(a1, a2, a3);
  v5 = *(_QWORD *)(a1 + 304);
  v6 = *(__m128i **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 0x10u )
  {
    sub_CB6200(v5, ", frame_ptr_rel, ", 0x11u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7F8A0);
    v6[1].m128i_i8[0] = 32;
    *v6 = si128;
    *(_QWORD *)(v5 + 32) += 17LL;
  }
  sub_CB59F0(*(_QWORD *)(a1 + 304), a4);
  v8 = *(_QWORD *)(a1 + 344);
  if ( v8 )
  {
    v12 = *(_QWORD *)(a1 + 304);
    v13 = *(unsigned __int8 **)(a1 + 336);
    v14 = *(_QWORD *)(a1 + 344);
    v15 = *(void **)(v12 + 32);
    if ( v8 > *(_QWORD *)(v12 + 24) - (_QWORD)v15 )
    {
      sub_CB6200(*(_QWORD *)(a1 + 304), v13, v14);
    }
    else
    {
      memcpy(v15, v13, v14);
      *(_QWORD *)(v12 + 32) += v8;
    }
  }
  v9 = *(_BYTE *)(a1 + 745) == 0;
  *(_QWORD *)(a1 + 344) = 0;
  if ( !v9 )
    return (_BYTE *)sub_E4D630((__int64 *)a1);
  v11 = *(_QWORD *)(a1 + 304);
  result = *(_BYTE **)(v11 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v11 + 24) )
    return (_BYTE *)sub_CB5D20(v11, 10);
  *(_QWORD *)(v11 + 32) = result + 1;
  *result = 10;
  return result;
}

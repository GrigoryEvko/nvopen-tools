// Function: sub_2AB80B0
// Address: 0x2ab80b0
//
_BYTE *__fastcall sub_2AB80B0(_BYTE *a1, __int64 a2, __int64 (__fastcall *a3)(__int64, char *, __int64), __int64 a4)
{
  __int64 v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // rsi
  unsigned __int64 v10; // rax
  size_t v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdi
  __m128i si128; // xmm0
  __int64 v15; // rax
  __int64 v16; // rdi
  __m128i v17; // xmm0
  _BYTE *result; // rax
  unsigned __int64 v19; // rax

  v6 = a3(a4, "LoopVectorizePass]", 17);
  v8 = *(_BYTE **)(a2 + 32);
  v9 = (unsigned __int8 *)v6;
  v10 = *(_QWORD *)(a2 + 24);
  v11 = v7;
  if ( v10 - (unsigned __int64)v8 < v7 )
  {
    sub_CB6200(a2, v9, v7);
    v8 = *(_BYTE **)(a2 + 32);
    v10 = *(_QWORD *)(a2 + 24);
  }
  else if ( v7 )
  {
    memcpy(v8, v9, v7);
    v19 = *(_QWORD *)(a2 + 24);
    v8 = (_BYTE *)(v11 + *(_QWORD *)(a2 + 32));
    *(_QWORD *)(a2 + 32) = v8;
    if ( v19 > (unsigned __int64)v8 )
      goto LABEL_4;
    goto LABEL_18;
  }
  if ( v10 > (unsigned __int64)v8 )
  {
LABEL_4:
    *(_QWORD *)(a2 + 32) = v8 + 1;
    *v8 = 60;
    goto LABEL_5;
  }
LABEL_18:
  sub_CB5D20(a2, 60);
LABEL_5:
  v12 = *(_QWORD *)(a2 + 32);
  v13 = a2;
  if ( !*a1 )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v12) > 2 )
    {
      *(_BYTE *)(v12 + 2) = 45;
      *(_WORD *)v12 = 28526;
      v12 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v12;
    }
    else
    {
      v13 = sub_CB6200(a2, "no-", 3u);
      v12 = *(_QWORD *)(v13 + 32);
    }
  }
  if ( (unsigned __int64)(*(_QWORD *)(v13 + 24) - v12) <= 0x16 )
  {
    sub_CB6200(v13, "interleave-forced-only;", 0x17u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_439F190);
    *(_DWORD *)(v12 + 16) = 1852779876;
    *(_WORD *)(v12 + 20) = 31084;
    *(_BYTE *)(v12 + 22) = 59;
    *(__m128i *)v12 = si128;
    *(_QWORD *)(v13 + 32) += 23LL;
  }
  v15 = *(_QWORD *)(a2 + 32);
  v16 = a2;
  if ( !a1[1] )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v15) > 2 )
    {
      *(_BYTE *)(v15 + 2) = 45;
      *(_WORD *)v15 = 28526;
      v15 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v15;
    }
    else
    {
      v16 = sub_CB6200(a2, "no-", 3u);
      v15 = *(_QWORD *)(v16 + 32);
    }
  }
  if ( (unsigned __int64)(*(_QWORD *)(v16 + 24) - v15) <= 0x15 )
  {
    sub_CB6200(v16, "vectorize-forced-only;", 0x16u);
  }
  else
  {
    v17 = _mm_load_si128((const __m128i *)&xmmword_439F1A0);
    *(_DWORD *)(v15 + 16) = 1819176749;
    *(_WORD *)(v15 + 20) = 15225;
    *(__m128i *)v15 = v17;
    *(_QWORD *)(v16 + 32) += 22LL;
  }
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 62);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 62;
  return result;
}

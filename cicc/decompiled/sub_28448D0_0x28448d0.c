// Function: sub_28448D0
// Address: 0x28448d0
//
_BYTE *__fastcall sub_28448D0(_BYTE *a1, __int64 a2, __int64 (__fastcall *a3)(__int64, char *, __int64), __int64 a4)
{
  __int64 v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // rsi
  _BYTE *v10; // rax
  size_t v11; // r13
  __int64 v12; // rdx
  __m128i si128; // xmm0
  _BYTE *v14; // rdx
  _BYTE *result; // rax
  _BYTE *v16; // rax

  v6 = a3(a4, "LoopRotatePass]", 14);
  v8 = *(_BYTE **)(a2 + 32);
  v9 = (unsigned __int8 *)v6;
  v10 = *(_BYTE **)(a2 + 24);
  v11 = v7;
  if ( v10 - v8 < v7 )
  {
    sub_CB6200(a2, v9, v7);
    v10 = *(_BYTE **)(a2 + 24);
    v8 = *(_BYTE **)(a2 + 32);
LABEL_3:
    if ( v8 != v10 )
      goto LABEL_4;
    goto LABEL_13;
  }
  if ( !v7 )
    goto LABEL_3;
  memcpy(v8, v9, v7);
  v16 = *(_BYTE **)(a2 + 24);
  v8 = (_BYTE *)(v11 + *(_QWORD *)(a2 + 32));
  *(_QWORD *)(a2 + 32) = v8;
  if ( v8 != v16 )
  {
LABEL_4:
    *v8 = 60;
    v12 = *(_QWORD *)(a2 + 32) + 1LL;
    *(_QWORD *)(a2 + 32) = v12;
    if ( *a1 )
      goto LABEL_5;
LABEL_14:
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v12) <= 2 )
    {
      sub_CB6200(a2, "no-", 3u);
      v12 = *(_QWORD *)(a2 + 32);
    }
    else
    {
      *(_BYTE *)(v12 + 2) = 45;
      *(_WORD *)v12 = 28526;
      v12 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v12;
    }
    goto LABEL_5;
  }
LABEL_13:
  sub_CB6200(a2, "<", 1u);
  v12 = *(_QWORD *)(a2 + 32);
  if ( !*a1 )
    goto LABEL_14;
LABEL_5:
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v12) <= 0x12 )
  {
    sub_CB6200(a2, "header-duplication;", 0x13u);
    v14 = *(_BYTE **)(a2 + 32);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4395AA0);
    *(_BYTE *)(v12 + 18) = 59;
    *(_WORD *)(v12 + 16) = 28271;
    *(__m128i *)v12 = si128;
    v14 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 19LL);
    *(_QWORD *)(a2 + 32) = v14;
  }
  if ( !a1[1] )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v14 <= 2u )
    {
      sub_CB6200(a2, "no-", 3u);
      v14 = *(_BYTE **)(a2 + 32);
    }
    else
    {
      v14[2] = 45;
      *(_WORD *)v14 = 28526;
      v14 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 3LL);
      *(_QWORD *)(a2 + 32) = v14;
    }
  }
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v14 <= 0xEu )
  {
    sub_CB6200(a2, (unsigned __int8 *)"prepare-for-lto", 0xFu);
    result = *(_BYTE **)(a2 + 32);
  }
  else
  {
    qmemcpy(v14, "prepare-for-lto", 15);
    result = (_BYTE *)(*(_QWORD *)(a2 + 32) + 15LL);
    *(_QWORD *)(a2 + 32) = result;
  }
  if ( *(_BYTE **)(a2 + 24) == result )
    return (_BYTE *)sub_CB6200(a2, (unsigned __int8 *)">", 1u);
  *result = 62;
  ++*(_QWORD *)(a2 + 32);
  return result;
}

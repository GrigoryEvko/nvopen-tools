// Function: sub_27EDB50
// Address: 0x27edb50
//
_BYTE *__fastcall sub_27EDB50(__int64 a1, __int64 a2, __int64 (__fastcall *a3)(__int64, char *, __int64), __int64 a4)
{
  __int64 v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // rsi
  unsigned __int64 v10; // rax
  size_t v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdi
  _BYTE *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __m128i si128; // xmm0
  _BYTE *result; // rax
  unsigned __int64 v19; // rax

  v6 = a3(a4, "LICMPass]", 8);
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
    goto LABEL_20;
  }
  if ( v10 > (unsigned __int64)v8 )
  {
LABEL_4:
    *(_QWORD *)(a2 + 32) = v8 + 1;
    *v8 = 60;
    goto LABEL_5;
  }
LABEL_20:
  sub_CB5D20(a2, 60);
LABEL_5:
  v12 = *(_QWORD *)(a2 + 32);
  v13 = a2;
  if ( !*(_BYTE *)(a1 + 8) )
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
  if ( (unsigned __int64)(*(_QWORD *)(v13 + 24) - v12) <= 0xF )
  {
    sub_CB6200(v13, "allowspeculation", 0x10u);
  }
  else
  {
    *(__m128i *)v12 = _mm_load_si128((const __m128i *)&xmmword_4394A60);
    *(_QWORD *)(v13 + 32) += 16LL;
  }
  v14 = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == v14 )
  {
    v15 = sub_CB6200(a2, (unsigned __int8 *)";", 1u);
    v16 = *(_QWORD *)(v15 + 32);
  }
  else
  {
    *v14 = 59;
    v15 = a2;
    v16 = *(_QWORD *)(a2 + 32) + 1LL;
    *(_QWORD *)(a2 + 32) = v16;
  }
  if ( !*(_BYTE *)(a1 + 9) )
  {
    if ( (unsigned __int64)(*(_QWORD *)(v15 + 24) - v16) > 2 )
    {
      *(_BYTE *)(v16 + 2) = 45;
      *(_WORD *)v16 = 28526;
      v16 = *(_QWORD *)(v15 + 32) + 3LL;
      *(_QWORD *)(v15 + 32) = v16;
    }
    else
    {
      v15 = sub_CB6200(v15, "no-", 3u);
      v16 = *(_QWORD *)(v15 + 32);
    }
  }
  if ( (unsigned __int64)(*(_QWORD *)(v15 + 24) - v16) <= 0x11 )
  {
    sub_CB6200(v15, "conservative-calls", 0x12u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4394A70);
    *(_WORD *)(v16 + 16) = 29548;
    *(__m128i *)v16 = si128;
    *(_QWORD *)(v15 + 32) += 18LL;
  }
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 62);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 62;
  return result;
}

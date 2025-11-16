// Function: sub_F10CC0
// Address: 0xf10cc0
//
_BYTE *__fastcall sub_F10CC0(__int64 a1, __int64 a2, __int64 (__fastcall *a3)(__int64, char *, __int64), __int64 a4)
{
  __int64 v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // rsi
  unsigned __int64 v10; // rax
  size_t v11; // r13
  void *v12; // rdx
  __int64 v13; // rdi
  __int64 v14; // rdi
  _BYTE *v15; // rax
  _BYTE *v16; // rax
  __int64 v17; // rdi
  _BYTE *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  __m128i si128; // xmm0
  _BYTE *result; // rax
  unsigned __int64 v23; // rax

  v6 = a3(a4, "InstCombinePass]", 15);
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
    v23 = *(_QWORD *)(a2 + 24);
    v8 = (_BYTE *)(v11 + *(_QWORD *)(a2 + 32));
    *(_QWORD *)(a2 + 32) = v8;
    if ( (unsigned __int64)v8 < v23 )
      goto LABEL_4;
    goto LABEL_24;
  }
  if ( (unsigned __int64)v8 < v10 )
  {
LABEL_4:
    *(_QWORD *)(a2 + 32) = v8 + 1;
    *v8 = 60;
    goto LABEL_5;
  }
LABEL_24:
  sub_CB5D20(a2, 60);
LABEL_5:
  v12 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v12 <= 0xEu )
  {
    v13 = sub_CB6200(a2, "max-iterations=", 0xFu);
  }
  else
  {
    v13 = a2;
    qmemcpy(v12, "max-iterations=", 15);
    *(_QWORD *)(a2 + 32) += 15LL;
  }
  v14 = sub_CB59D0(v13, *(unsigned int *)(a1 + 2276));
  v15 = *(_BYTE **)(v14 + 32);
  if ( *(_BYTE **)(v14 + 24) == v15 )
  {
    sub_CB6200(v14, (unsigned __int8 *)";", 1u);
  }
  else
  {
    *v15 = 59;
    ++*(_QWORD *)(v14 + 32);
  }
  v16 = *(_BYTE **)(a2 + 32);
  v17 = a2;
  if ( !*(_BYTE *)(a1 + 2272) )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v16 > 2u )
    {
      v16[2] = 45;
      *(_WORD *)v16 = 28526;
      v16 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 3LL);
      *(_QWORD *)(a2 + 32) = v16;
    }
    else
    {
      v17 = sub_CB6200(a2, "no-", 3u);
      v16 = *(_BYTE **)(v17 + 32);
    }
  }
  if ( *(_QWORD *)(v17 + 24) - (_QWORD)v16 <= 0xEu )
  {
    sub_CB6200(v17, "verify-fixpoint", 0xFu);
  }
  else
  {
    qmemcpy(v16, "verify-fixpoint", 15);
    *(_QWORD *)(v17 + 32) += 15LL;
  }
  v18 = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == v18 )
  {
    v19 = sub_CB6200(a2, (unsigned __int8 *)";", 1u);
    v20 = *(_QWORD *)(v19 + 32);
  }
  else
  {
    *v18 = 59;
    v19 = a2;
    v20 = *(_QWORD *)(a2 + 32) + 1LL;
    *(_QWORD *)(a2 + 32) = v20;
  }
  if ( !*(_BYTE *)(a1 + 2280) )
  {
    if ( (unsigned __int64)(*(_QWORD *)(v19 + 24) - v20) > 2 )
    {
      *(_BYTE *)(v20 + 2) = 45;
      *(_WORD *)v20 = 28526;
      v20 = *(_QWORD *)(v19 + 32) + 3LL;
      *(_QWORD *)(v19 + 32) = v20;
    }
    else
    {
      v19 = sub_CB6200(v19, "no-", 3u);
      v20 = *(_QWORD *)(v19 + 32);
    }
  }
  if ( (unsigned __int64)(*(_QWORD *)(v19 + 24) - v20) <= 0x1D )
  {
    sub_CB6200(v19, "aggressive-aggregate-splitting", 0x1Eu);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F89990);
    qmemcpy((void *)(v20 + 16), "gate-splitting", 14);
    *(__m128i *)v20 = si128;
    *(_QWORD *)(v19 + 32) += 30LL;
  }
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 62);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 62;
  return result;
}

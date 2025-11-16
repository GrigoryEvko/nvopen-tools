// Function: sub_2E60D30
// Address: 0x2e60d30
//
__int64 *__fastcall sub_2E60D30(__int64 a1, __int64 a2)
{
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  __int64 v6; // r13
  __int64 v7; // rax
  size_t v8; // rdx
  _BYTE *v9; // rdi
  unsigned __int8 *v10; // rsi
  _BYTE *v11; // rax
  size_t v12; // r14
  _BYTE *v14; // rax

  v3 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 <= 0x1Eu )
  {
    v6 = sub_CB6200(a2, "MachineCycleInfo for function: ", 0x1Fu);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4450200);
    v6 = a2;
    qmemcpy(&v3[1], " for function: ", 15);
    *v3 = si128;
    *(_QWORD *)(a2 + 32) += 31LL;
  }
  v7 = sub_2E791E0(*(_QWORD *)(a1 + 200));
  v9 = *(_BYTE **)(v6 + 32);
  v10 = (unsigned __int8 *)v7;
  v11 = *(_BYTE **)(v6 + 24);
  v12 = v8;
  if ( v11 - v9 < v8 )
  {
    v6 = sub_CB6200(v6, v10, v8);
    v11 = *(_BYTE **)(v6 + 24);
    v9 = *(_BYTE **)(v6 + 32);
  }
  else if ( v8 )
  {
    memcpy(v9, v10, v8);
    v14 = *(_BYTE **)(v6 + 24);
    v9 = (_BYTE *)(v12 + *(_QWORD *)(v6 + 32));
    *(_QWORD *)(v6 + 32) = v9;
    if ( v14 != v9 )
      goto LABEL_6;
LABEL_9:
    sub_CB6200(v6, (unsigned __int8 *)"\n", 1u);
    return sub_2E60390(a1 + 208, a2);
  }
  if ( v11 == v9 )
    goto LABEL_9;
LABEL_6:
  *v9 = 10;
  ++*(_QWORD *)(v6 + 32);
  return sub_2E60390(a1 + 208, a2);
}

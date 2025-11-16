// Function: sub_11FC210
// Address: 0x11fc210
//
__int64 *__fastcall sub_11FC210(__int64 a1, __int64 a2)
{
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  __int64 v6; // r13
  const char *v7; // rax
  size_t v8; // rdx
  _BYTE *v9; // rdi
  unsigned __int8 *v10; // rsi
  _BYTE *v11; // rax
  size_t v12; // r14
  _BYTE *v14; // rax

  v3 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 <= 0x17u )
  {
    v6 = sub_CB6200(a2, "CycleInfo for function: ", 0x18u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F95610);
    v6 = a2;
    v3[1].m128i_i64[0] = 0x203A6E6F6974636ELL;
    *v3 = si128;
    *(_QWORD *)(a2 + 32) += 24LL;
  }
  v7 = sub_BD5D20(*(_QWORD *)(a1 + 176));
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
    if ( v9 != v14 )
      goto LABEL_6;
LABEL_9:
    sub_CB6200(v6, (unsigned __int8 *)"\n", 1u);
    return sub_E39A30(a1 + 184, a2);
  }
  if ( v9 == v11 )
    goto LABEL_9;
LABEL_6:
  *v9 = 10;
  ++*(_QWORD *)(v6 + 32);
  return sub_E39A30(a1 + 184, a2);
}

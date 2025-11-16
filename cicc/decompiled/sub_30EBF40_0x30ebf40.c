// Function: sub_30EBF40
// Address: 0x30ebf40
//
__int64 __fastcall sub_30EBF40(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r13
  __m128i *v8; // rdx
  const char *v9; // rax
  size_t v10; // rdx
  _WORD *v11; // rdi
  unsigned __int8 *v12; // rsi
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  _BYTE *v15; // rax
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  size_t v19; // [rsp+8h] [rbp-28h]

  v7 = *a2;
  v8 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v8 <= 0x2Fu )
  {
    v7 = sub_CB6200(*a2, "[InlineSizeEstimatorAnalysis] size estimate for ", 0x30u);
  }
  else
  {
    *v8 = _mm_load_si128((const __m128i *)&xmmword_44CDED0);
    v8[1] = _mm_load_si128((const __m128i *)&xmmword_44CDEE0);
    v8[2] = _mm_load_si128((const __m128i *)&xmmword_44CDEF0);
    *(_QWORD *)(v7 + 32) += 48LL;
  }
  v9 = sub_BD5D20(a3);
  v11 = *(_WORD **)(v7 + 32);
  v12 = (unsigned __int8 *)v9;
  v13 = *(_QWORD *)(v7 + 24) - (_QWORD)v11;
  if ( v13 < v10 )
  {
    v18 = sub_CB6200(v7, v12, v10);
    v11 = *(_WORD **)(v18 + 32);
    v7 = v18;
    v13 = *(_QWORD *)(v18 + 24) - (_QWORD)v11;
  }
  else if ( v10 )
  {
    v19 = v10;
    memcpy(v11, v12, v10);
    v11 = (_WORD *)(v19 + *(_QWORD *)(v7 + 32));
    v17 = *(_QWORD *)(v7 + 24) - (_QWORD)v11;
    *(_QWORD *)(v7 + 32) = v11;
    if ( v17 > 1 )
      goto LABEL_6;
    goto LABEL_14;
  }
  if ( v13 > 1 )
  {
LABEL_6:
    *v11 = 8250;
    *(_QWORD *)(v7 + 32) += 2LL;
    goto LABEL_7;
  }
LABEL_14:
  v7 = sub_CB6200(v7, (unsigned __int8 *)": ", 2u);
LABEL_7:
  v14 = sub_BC1CD0(a4, &unk_5031220, a3);
  if ( !*(_BYTE *)(v14 + 16) )
  {
    sub_F03F40(v7);
    v15 = *(_BYTE **)(v7 + 32);
    if ( *(_BYTE **)(v7 + 24) != v15 )
      goto LABEL_9;
LABEL_12:
    sub_CB6200(v7, (unsigned __int8 *)"\n", 1u);
    goto LABEL_10;
  }
  sub_CB59D0(v7, *(_QWORD *)(v14 + 8));
  v15 = *(_BYTE **)(v7 + 32);
  if ( *(_BYTE **)(v7 + 24) == v15 )
    goto LABEL_12;
LABEL_9:
  *v15 = 10;
  ++*(_QWORD *)(v7 + 32);
LABEL_10:
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}

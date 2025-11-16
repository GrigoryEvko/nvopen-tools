// Function: sub_30C5D10
// Address: 0x30c5d10
//
__int64 __fastcall sub_30C5D10(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v8; // r13
  __m128i *v9; // rdx
  __m128i si128; // xmm0
  _BYTE *v11; // rax
  const char *v12; // rax
  size_t v13; // rdx
  _WORD *v14; // rdi
  unsigned __int8 *v15; // rsi
  unsigned __int64 v16; // rax
  _BYTE *v17; // rax
  __int64 v18; // rax
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  size_t v22; // [rsp+8h] [rbp-38h]

  v8 = *a2;
  v9 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v9 <= 0x2Du )
  {
    v8 = sub_CB6200(*a2, "Printing analysis results of CFA for function ", 0x2Eu);
    v11 = *(_BYTE **)(v8 + 32);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F70AE0);
    qmemcpy(&v9[2], " for function ", 14);
    *v9 = si128;
    v9[1] = _mm_load_si128((const __m128i *)&xmmword_44CC630);
    v11 = (_BYTE *)(*(_QWORD *)(v8 + 32) + 46LL);
    *(_QWORD *)(v8 + 32) = v11;
  }
  if ( *(_BYTE **)(v8 + 24) == v11 )
  {
    v8 = sub_CB6200(v8, (unsigned __int8 *)"'", 1u);
  }
  else
  {
    *v11 = 39;
    ++*(_QWORD *)(v8 + 32);
  }
  v12 = sub_BD5D20(a3);
  v14 = *(_WORD **)(v8 + 32);
  v15 = (unsigned __int8 *)v12;
  v16 = *(_QWORD *)(v8 + 24) - (_QWORD)v14;
  if ( v16 < v13 )
  {
    v21 = sub_CB6200(v8, v15, v13);
    v14 = *(_WORD **)(v21 + 32);
    v8 = v21;
    v16 = *(_QWORD *)(v21 + 24) - (_QWORD)v14;
LABEL_7:
    if ( v16 > 1 )
      goto LABEL_8;
LABEL_12:
    v8 = sub_CB6200(v8, "':", 2u);
    v17 = *(_BYTE **)(v8 + 32);
    if ( *(_BYTE **)(v8 + 24) != v17 )
      goto LABEL_9;
LABEL_13:
    sub_CB6200(v8, (unsigned __int8 *)"\n", 1u);
    goto LABEL_10;
  }
  if ( !v13 )
    goto LABEL_7;
  v22 = v13;
  memcpy(v14, v15, v13);
  v14 = (_WORD *)(v22 + *(_QWORD *)(v8 + 32));
  v20 = *(_QWORD *)(v8 + 24) - (_QWORD)v14;
  *(_QWORD *)(v8 + 32) = v14;
  if ( v20 <= 1 )
    goto LABEL_12;
LABEL_8:
  *v14 = 14887;
  v17 = (_BYTE *)(*(_QWORD *)(v8 + 32) + 2LL);
  *(_QWORD *)(v8 + 32) = v17;
  if ( *(_BYTE **)(v8 + 24) == v17 )
    goto LABEL_13;
LABEL_9:
  *v17 = 10;
  ++*(_QWORD *)(v8 + 32);
LABEL_10:
  v18 = sub_BC1CD0(a4, &unk_502ED90, a3);
  sub_30C4110((signed __int64 *)(v18 + 8), *a2);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}

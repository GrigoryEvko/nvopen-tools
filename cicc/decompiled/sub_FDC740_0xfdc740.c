// Function: sub_FDC740
// Address: 0xfdc740
//
__int64 __fastcall sub_FDC740(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r13
  __m128i *v8; // rdx
  __m128i si128; // xmm0
  _BYTE *v10; // rax
  const char *v11; // rax
  size_t v12; // rdx
  _WORD *v13; // rdi
  unsigned __int8 *v14; // rsi
  unsigned __int64 v15; // rax
  _BYTE *v16; // rax
  __int64 v17; // rax
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  size_t v21; // [rsp+8h] [rbp-38h]

  v7 = *a2;
  v8 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v8 <= 0x2Du )
  {
    v7 = sub_CB6200(*a2, "Printing analysis results of BFI for function ", 0x2Eu);
    v10 = *(_BYTE **)(v7 + 32);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F70AE0);
    qmemcpy(&v8[2], " for function ", 14);
    *v8 = si128;
    v8[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CAD0);
    v10 = (_BYTE *)(*(_QWORD *)(v7 + 32) + 46LL);
    *(_QWORD *)(v7 + 32) = v10;
  }
  if ( *(_BYTE **)(v7 + 24) == v10 )
  {
    v7 = sub_CB6200(v7, (unsigned __int8 *)"'", 1u);
  }
  else
  {
    *v10 = 39;
    ++*(_QWORD *)(v7 + 32);
  }
  v11 = sub_BD5D20(a3);
  v13 = *(_WORD **)(v7 + 32);
  v14 = (unsigned __int8 *)v11;
  v15 = *(_QWORD *)(v7 + 24) - (_QWORD)v13;
  if ( v15 < v12 )
  {
    v20 = sub_CB6200(v7, v14, v12);
    v13 = *(_WORD **)(v20 + 32);
    v7 = v20;
    v15 = *(_QWORD *)(v20 + 24) - (_QWORD)v13;
LABEL_7:
    if ( v15 > 1 )
      goto LABEL_8;
LABEL_12:
    v7 = sub_CB6200(v7, "':", 2u);
    v16 = *(_BYTE **)(v7 + 32);
    if ( *(_BYTE **)(v7 + 24) != v16 )
      goto LABEL_9;
LABEL_13:
    sub_CB6200(v7, (unsigned __int8 *)"\n", 1u);
    goto LABEL_10;
  }
  if ( !v12 )
    goto LABEL_7;
  v21 = v12;
  memcpy(v13, v14, v12);
  v13 = (_WORD *)(v21 + *(_QWORD *)(v7 + 32));
  v19 = *(_QWORD *)(v7 + 24) - (_QWORD)v13;
  *(_QWORD *)(v7 + 32) = v13;
  if ( v19 <= 1 )
    goto LABEL_12;
LABEL_8:
  *v13 = 14887;
  v16 = (_BYTE *)(*(_QWORD *)(v7 + 32) + 2LL);
  *(_QWORD *)(v7 + 32) = v16;
  if ( *(_BYTE **)(v7 + 24) == v16 )
    goto LABEL_13;
LABEL_9:
  *v16 = 10;
  ++*(_QWORD *)(v7 + 32);
LABEL_10:
  v17 = sub_BC1CD0(a4, &unk_4F8D9A8, a3);
  sub_FDC540((__int64 *)(v17 + 8));
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}

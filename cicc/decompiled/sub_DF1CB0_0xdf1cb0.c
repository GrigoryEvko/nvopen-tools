// Function: sub_DF1CB0
// Address: 0xdf1cb0
//
__int64 __fastcall sub_DF1CB0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v8; // r13
  __m128i *v9; // rdx
  __m128i si128; // xmm0
  const char *v11; // rax
  size_t v12; // rdx
  _BYTE *v13; // rdi
  unsigned __int8 *v14; // rsi
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  size_t v20; // [rsp+8h] [rbp-38h]

  v8 = *a2;
  v9 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v9 <= 0x3Bu )
  {
    v8 = sub_CB6200(*a2, "Printing analysis 'Scalar Evolution Analysis' for function '", 0x3Cu);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F70AE0);
    qmemcpy(&v9[3], "r function '", 12);
    *v9 = si128;
    v9[1] = _mm_load_si128((const __m128i *)&xmmword_3F74F20);
    v9[2] = _mm_load_si128((const __m128i *)&xmmword_3F74F30);
    *(_QWORD *)(v8 + 32) += 60LL;
  }
  v11 = sub_BD5D20(a3);
  v13 = *(_BYTE **)(v8 + 32);
  v14 = (unsigned __int8 *)v11;
  v15 = *(_QWORD *)(v8 + 24) - (_QWORD)v13;
  if ( v15 < v12 )
  {
    v19 = sub_CB6200(v8, v14, v12);
    v13 = *(_BYTE **)(v19 + 32);
    v8 = v19;
    v15 = *(_QWORD *)(v19 + 24) - (_QWORD)v13;
  }
  else if ( v12 )
  {
    v20 = v12;
    memcpy(v13, v14, v12);
    v13 = (_BYTE *)(v20 + *(_QWORD *)(v8 + 32));
    v18 = *(_QWORD *)(v8 + 24) - (_QWORD)v13;
    *(_QWORD *)(v8 + 32) = v13;
    if ( v18 > 2 )
      goto LABEL_6;
LABEL_9:
    sub_CB6200(v8, "':\n", 3u);
    goto LABEL_7;
  }
  if ( v15 <= 2 )
    goto LABEL_9;
LABEL_6:
  v13[2] = 10;
  *(_WORD *)v13 = 14887;
  *(_QWORD *)(v8 + 32) += 3LL;
LABEL_7:
  v16 = sub_BC1CD0(a4, &unk_4F881D0, a3);
  sub_DF0A60(v16 + 8, *a2);
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

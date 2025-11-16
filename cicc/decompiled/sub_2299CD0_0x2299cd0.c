// Function: sub_2299CD0
// Address: 0x2299cd0
//
__int64 __fastcall sub_2299CD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // r13
  __m128i *v9; // rdx
  __m128i si128; // xmm0
  const char *v11; // rax
  size_t v12; // rdx
  _BYTE *v13; // rdi
  unsigned __int8 *v14; // rsi
  unsigned __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // rax
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  char v21; // [rsp+8h] [rbp-38h]
  size_t v22; // [rsp+8h] [rbp-38h]

  v8 = *(_QWORD *)a2;
  v9 = *(__m128i **)(*(_QWORD *)a2 + 32LL);
  if ( *(_QWORD *)(*(_QWORD *)a2 + 24LL) - (_QWORD)v9 <= 0x35u )
  {
    v8 = sub_CB6200(*(_QWORD *)a2, "Printing analysis 'Dependence Analysis' for function '", 0x36u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F70AE0);
    v9[3].m128i_i32[0] = 1852795252;
    v9[3].m128i_i16[2] = 10016;
    *v9 = si128;
    v9[1] = _mm_load_si128((const __m128i *)&xmmword_4366100);
    v9[2] = _mm_load_si128((const __m128i *)&xmmword_4366110);
    *(_QWORD *)(v8 + 32) += 54LL;
  }
  v11 = sub_BD5D20(a3);
  v13 = *(_BYTE **)(v8 + 32);
  v14 = (unsigned __int8 *)v11;
  v15 = *(_QWORD *)(v8 + 24) - (_QWORD)v13;
  if ( v15 < v12 )
  {
    v20 = sub_CB6200(v8, v14, v12);
    v13 = *(_BYTE **)(v20 + 32);
    v8 = v20;
    v15 = *(_QWORD *)(v20 + 24) - (_QWORD)v13;
  }
  else if ( v12 )
  {
    v22 = v12;
    memcpy(v13, v14, v12);
    v13 = (_BYTE *)(v22 + *(_QWORD *)(v8 + 32));
    v19 = *(_QWORD *)(v8 + 24) - (_QWORD)v13;
    *(_QWORD *)(v8 + 32) = v13;
    if ( v19 > 2 )
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
  v21 = *(_BYTE *)(a2 + 8);
  v16 = sub_BC1CD0(a4, &unk_4F881D0, a3);
  v17 = sub_BC1CD0(a4, &unk_4FDB350, a3);
  sub_2299720(*(_QWORD *)a2, v17 + 8, v16 + 8, v21);
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

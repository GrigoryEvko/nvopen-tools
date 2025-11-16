// Function: sub_2E39CA0
// Address: 0x2e39ca0
//
__int64 __fastcall sub_2E39CA0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r13
  _QWORD *v8; // r15
  __m128i *v9; // rdx
  __m128i si128; // xmm0
  __int64 v11; // rax
  size_t v12; // rdx
  _BYTE *v13; // rdi
  unsigned __int8 *v14; // rsi
  unsigned __int64 v15; // rax
  size_t v16; // r14
  unsigned __int64 v18; // rax
  __int64 v19; // rax

  v6 = sub_2EB2140(a4, &unk_501EC10);
  v7 = *a2;
  v8 = (_QWORD *)(v6 + 8);
  v9 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v9 <= 0x2Du )
  {
    v7 = sub_CB6200(*a2, "Machine block frequency for machine function: ", 0x2Eu);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_444FFC0);
    qmemcpy(&v9[2], "ine function: ", 14);
    *v9 = si128;
    v9[1] = _mm_load_si128((const __m128i *)&xmmword_444FFD0);
    *(_QWORD *)(v7 + 32) += 46LL;
  }
  v11 = sub_2E791E0(a3);
  v13 = *(_BYTE **)(v7 + 32);
  v14 = (unsigned __int8 *)v11;
  v15 = *(_QWORD *)(v7 + 24);
  v16 = v12;
  if ( v15 - (unsigned __int64)v13 < v12 )
  {
    v19 = sub_CB6200(v7, v14, v12);
    v13 = *(_BYTE **)(v19 + 32);
    v7 = v19;
    v15 = *(_QWORD *)(v19 + 24);
  }
  else if ( v12 )
  {
    memcpy(v13, v14, v12);
    v18 = *(_QWORD *)(v7 + 24);
    v13 = (_BYTE *)(v16 + *(_QWORD *)(v7 + 32));
    *(_QWORD *)(v7 + 32) = v13;
    if ( v18 > (unsigned __int64)v13 )
      goto LABEL_6;
LABEL_9:
    sub_CB5D20(v7, 10);
    goto LABEL_7;
  }
  if ( v15 <= (unsigned __int64)v13 )
    goto LABEL_9;
LABEL_6:
  *(_QWORD *)(v7 + 32) = v13 + 1;
  *v13 = 10;
LABEL_7:
  sub_2E39C90(v8);
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

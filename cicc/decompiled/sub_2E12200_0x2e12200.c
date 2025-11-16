// Function: sub_2E12200
// Address: 0x2e12200
//
__int64 __fastcall sub_2E12200(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v8; // r13
  __m128i *v9; // rdx
  __m128i si128; // xmm0
  __int64 v11; // rax
  size_t v12; // rdx
  _WORD *v13; // rdi
  unsigned __int8 *v14; // rsi
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  size_t v20; // [rsp+8h] [rbp-38h]

  v8 = *a2;
  v9 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v9 <= 0x24u )
  {
    v8 = sub_CB6200(*a2, "Live intervals for machine function: ", 0x25u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_444FD60);
    v9[2].m128i_i32[0] = 980316009;
    v9[2].m128i_i8[4] = 32;
    *v9 = si128;
    v9[1] = _mm_load_si128((const __m128i *)&xmmword_444FD70);
    *(_QWORD *)(v8 + 32) += 37LL;
  }
  v11 = sub_2E791E0(a3);
  v13 = *(_WORD **)(v8 + 32);
  v14 = (unsigned __int8 *)v11;
  v15 = *(_QWORD *)(v8 + 24) - (_QWORD)v13;
  if ( v15 < v12 )
  {
    v19 = sub_CB6200(v8, v14, v12);
    v13 = *(_WORD **)(v19 + 32);
    v8 = v19;
    v15 = *(_QWORD *)(v19 + 24) - (_QWORD)v13;
  }
  else if ( v12 )
  {
    v20 = v12;
    memcpy(v13, v14, v12);
    v13 = (_WORD *)(v20 + *(_QWORD *)(v8 + 32));
    v18 = *(_QWORD *)(v8 + 24) - (_QWORD)v13;
    *(_QWORD *)(v8 + 32) = v13;
    if ( v18 > 1 )
      goto LABEL_6;
LABEL_9:
    sub_CB6200(v8, (unsigned __int8 *)":\n", 2u);
    goto LABEL_7;
  }
  if ( v15 <= 1 )
    goto LABEL_9;
LABEL_6:
  *v13 = 2618;
  *(_QWORD *)(v8 + 32) += 2LL;
LABEL_7:
  v16 = sub_2EB2140(a4, &unk_501EAD0);
  sub_2E11F00(v16 + 8, *a2);
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

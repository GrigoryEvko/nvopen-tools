// Function: sub_2C767C0
// Address: 0x2c767c0
//
__int64 __fastcall sub_2C767C0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // r13
  __m128i *v6; // rdx
  __m128i si128; // xmm0
  const char *v8; // rax
  size_t v9; // rdx
  _BYTE *v10; // rdi
  unsigned __int8 *v11; // rsi
  unsigned __int64 v12; // rax
  __int64 v13; // rdi
  void *v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v18; // rdi
  _WORD *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  _BYTE *v22; // rdx
  size_t v23; // [rsp+8h] [rbp-28h]

  sub_2C763F0(a3, *(_QWORD *)(a1 + 24));
  if ( (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
  {
    v5 = *(_QWORD *)(a1 + 24);
    v6 = *(__m128i **)(v5 + 32);
    if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 0x12u )
    {
      v5 = sub_CB6200(*(_QWORD *)(a1 + 24), ": Global Variable `", 0x13u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42D04F0);
      v6[1].m128i_i8[2] = 96;
      v6[1].m128i_i16[0] = 8293;
      *v6 = si128;
      *(_QWORD *)(v5 + 32) += 19LL;
    }
    v8 = sub_BD5D20(a2);
    v10 = *(_BYTE **)(v5 + 32);
    v11 = (unsigned __int8 *)v8;
    v12 = *(_QWORD *)(v5 + 24) - (_QWORD)v10;
    if ( v12 < v9 )
    {
      v20 = sub_CB6200(v5, v11, v9);
      v10 = *(_BYTE **)(v20 + 32);
      v5 = v20;
      v12 = *(_QWORD *)(v20 + 24) - (_QWORD)v10;
    }
    else if ( v9 )
    {
      v23 = v9;
      memcpy(v10, v11, v9);
      v21 = *(_QWORD *)(v5 + 24);
      v22 = (_BYTE *)(*(_QWORD *)(v5 + 32) + v23);
      *(_QWORD *)(v5 + 32) = v22;
      v10 = v22;
      v12 = v21 - (_QWORD)v22;
    }
    if ( v12 <= 2 )
    {
      sub_CB6200(v5, "': ", 3u);
    }
    else
    {
      v10[2] = 32;
      *(_WORD *)v10 = 14887;
      *(_QWORD *)(v5 + 32) += 3LL;
    }
  }
  else
  {
    v18 = *(_QWORD *)(a1 + 24);
    v19 = *(_WORD **)(v18 + 32);
    if ( *(_QWORD *)(v18 + 24) - (_QWORD)v19 <= 1u )
    {
      sub_CB6200(v18, (unsigned __int8 *)": ", 2u);
    }
    else
    {
      *v19 = 8250;
      *(_QWORD *)(v18 + 32) += 2LL;
    }
  }
  v13 = *(_QWORD *)(a1 + 24);
  v14 = *(void **)(v13 + 32);
  if ( *(_QWORD *)(v13 + 24) - (_QWORD)v14 <= 0xBu )
  {
    sub_CB6200(v13, "\n  context: ", 0xCu);
  }
  else
  {
    qmemcpy(v14, "\n  context: ", 12);
    *(_QWORD *)(v13 + 32) += 12LL;
  }
  sub_A69870(a2, *(_BYTE **)(a1 + 24), 0);
  v15 = *(_QWORD *)(a1 + 24);
  v16 = *(_QWORD *)(v15 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v15 + 24) - v16) <= 2 )
  {
    sub_CB6200(v15, "\n  ", 3u);
  }
  else
  {
    *(_BYTE *)(v16 + 2) = 32;
    *(_WORD *)v16 = 8202;
    *(_QWORD *)(v15 + 32) += 3LL;
  }
  return *(_QWORD *)(a1 + 24);
}

// Function: sub_1C31CC0
// Address: 0x1c31cc0
//
__int64 *__fastcall sub_1C31CC0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // r13
  __m128i *v5; // rdx
  const char *v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  char *v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v12; // rdi
  _WORD *v13; // rdx
  __int64 v14; // rax
  _BYTE *v15; // rdx
  __int64 v16; // rax
  size_t v17; // [rsp+8h] [rbp-28h]

  if ( a3 == 2 && !byte_4FBA540 )
    return sub_16E8D30();
  sub_1C31A90(a3, *(_QWORD *)(a1 + 24));
  if ( (*(_BYTE *)(a2 + 23) & 0x20) != 0 )
  {
    v4 = *(_QWORD *)(a1 + 24);
    v5 = *(__m128i **)(v4 + 24);
    if ( *(_QWORD *)(v4 + 16) - (_QWORD)v5 <= 0xFu )
    {
      v4 = sub_16E7EE0(*(_QWORD *)(a1 + 24), ": Global Value `", 0x10u);
    }
    else
    {
      *v5 = _mm_load_si128((const __m128i *)&xmmword_42D04E0);
      *(_QWORD *)(v4 + 24) += 16LL;
    }
    v6 = sub_1649960(a2);
    v8 = *(_BYTE **)(v4 + 24);
    v9 = (char *)v6;
    v10 = *(_QWORD *)(v4 + 16) - (_QWORD)v8;
    if ( v10 < v7 )
    {
      v16 = sub_16E7EE0(v4, v9, v7);
      v8 = *(_BYTE **)(v16 + 24);
      v4 = v16;
      if ( *(_QWORD *)(v16 + 16) - (_QWORD)v8 > 2u )
      {
LABEL_10:
        v8[2] = 32;
        *(_WORD *)v8 = 14887;
        *(_QWORD *)(v4 + 24) += 3LL;
        return *(__int64 **)(a1 + 24);
      }
    }
    else
    {
      if ( v7 )
      {
        v17 = v7;
        memcpy(v8, v9, v7);
        v14 = *(_QWORD *)(v4 + 16);
        v15 = (_BYTE *)(*(_QWORD *)(v4 + 24) + v17);
        *(_QWORD *)(v4 + 24) = v15;
        v8 = v15;
        v10 = v14 - (_QWORD)v15;
      }
      if ( v10 > 2 )
        goto LABEL_10;
    }
    sub_16E7EE0(v4, "': ", 3u);
    return *(__int64 **)(a1 + 24);
  }
  else
  {
    v12 = *(_QWORD *)(a1 + 24);
    v13 = *(_WORD **)(v12 + 24);
    if ( *(_QWORD *)(v12 + 16) - (_QWORD)v13 > 1u )
    {
      *v13 = 8250;
      *(_QWORD *)(v12 + 24) += 2LL;
      return *(__int64 **)(a1 + 24);
    }
    sub_16E7EE0(v12, ": ", 2u);
    return *(__int64 **)(a1 + 24);
  }
}

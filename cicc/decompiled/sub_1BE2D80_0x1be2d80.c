// Function: sub_1BE2D80
// Address: 0x1be2d80
//
__int64 __fastcall sub_1BE2D80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // r13
  __m128i *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rdx

  v4 = a2;
  v5 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v5) <= 2 )
  {
    v6 = sub_16E7EE0(a2, " +\n", 3u);
  }
  else
  {
    *(_BYTE *)(v5 + 2) = 10;
    v6 = a2;
    *(_WORD *)v5 = 11040;
    *(_QWORD *)(a2 + 24) += 3LL;
  }
  sub_16E2CE0(a3, v6);
  v7 = *(__m128i **)(v6 + 24);
  if ( *(_QWORD *)(v6 + 16) - (_QWORD)v7 <= 0xFu )
  {
    sub_16E7EE0(v6, "\"WIDEN-INDUCTION", 0x10u);
  }
  else
  {
    *v7 = _mm_load_si128((const __m128i *)&xmmword_42CAA90);
    *(_QWORD *)(v6 + 24) += 16LL;
  }
  v8 = *(_QWORD *)(a2 + 16);
  v9 = *(_QWORD *)(a2 + 24);
  if ( *(_QWORD *)(a1 + 48) )
  {
    if ( (unsigned __int64)(v8 - v9) <= 2 )
    {
      sub_16E7EE0(a2, "\\l\"", 3u);
      v10 = *(_QWORD *)(a2 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v10) <= 2 )
        goto LABEL_8;
    }
    else
    {
      *(_BYTE *)(v9 + 2) = 34;
      *(_WORD *)v9 = 27740;
      v10 = *(_QWORD *)(a2 + 24) + 3LL;
      v11 = *(_QWORD *)(a2 + 16);
      *(_QWORD *)(a2 + 24) = v10;
      if ( (unsigned __int64)(v11 - v10) <= 2 )
      {
LABEL_8:
        v12 = sub_16E7EE0(a2, " +\n", 3u);
LABEL_9:
        sub_16E2CE0(a3, v12);
        v13 = *(_QWORD *)(v12 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(v12 + 16) - v13) <= 2 )
        {
          v12 = sub_16E7EE0(v12, "\"  ", 3u);
        }
        else
        {
          *(_BYTE *)(v13 + 2) = 32;
          *(_WORD *)v13 = 8226;
          *(_QWORD *)(v12 + 24) += 3LL;
        }
        sub_1BE27E0(v12, *(_QWORD *)(a1 + 40));
        v14 = *(_QWORD *)(v12 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(v12 + 16) - v14) <= 2 )
        {
          sub_16E7EE0(v12, "\\l\"", 3u);
        }
        else
        {
          *(_BYTE *)(v14 + 2) = 34;
          *(_WORD *)v14 = 27740;
          *(_QWORD *)(v12 + 24) += 3LL;
        }
        v15 = *(_QWORD *)(a2 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v15) <= 2 )
        {
          v4 = sub_16E7EE0(a2, " +\n", 3u);
        }
        else
        {
          *(_BYTE *)(v15 + 2) = 10;
          *(_WORD *)v15 = 11040;
          *(_QWORD *)(a2 + 24) += 3LL;
        }
        sub_16E2CE0(a3, v4);
        v16 = *(_QWORD *)(v4 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(v4 + 16) - v16) <= 2 )
        {
          v4 = sub_16E7EE0(v4, "\"  ", 3u);
        }
        else
        {
          *(_BYTE *)(v16 + 2) = 32;
          *(_WORD *)v16 = 8226;
          *(_QWORD *)(v4 + 24) += 3LL;
        }
        v17 = *(_QWORD *)(a1 + 48);
        goto LABEL_18;
      }
    }
    *(_BYTE *)(v10 + 2) = 10;
    v12 = a2;
    *(_WORD *)v10 = 11040;
    *(_QWORD *)(a2 + 24) += 3LL;
    goto LABEL_9;
  }
  if ( v8 == v9 )
  {
    v4 = sub_16E7EE0(a2, " ", 1u);
  }
  else
  {
    *(_BYTE *)v9 = 32;
    ++*(_QWORD *)(a2 + 24);
  }
  v17 = *(_QWORD *)(a1 + 40);
LABEL_18:
  sub_1BE27E0(v4, v17);
  v18 = *(_QWORD *)(v4 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v4 + 16) - v18) <= 2 )
    return sub_16E7EE0(v4, "\\l\"", 3u);
  *(_BYTE *)(v18 + 2) = 34;
  *(_WORD *)v18 = 27740;
  *(_QWORD *)(v4 + 24) += 3LL;
  return 27740;
}

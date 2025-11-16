// Function: sub_1B903A0
// Address: 0x1b903a0
//
__int64 __fastcall sub_1B903A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // r13
  __m128i *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx

  v4 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v4) <= 2 )
  {
    v5 = sub_16E7EE0(a2, " +\n", 3u);
  }
  else
  {
    *(_BYTE *)(v4 + 2) = 10;
    v5 = a2;
    *(_WORD *)v4 = 11040;
    *(_QWORD *)(a2 + 24) += 3LL;
  }
  sub_16E2CE0(a3, v5);
  v6 = *(__m128i **)(v5 + 24);
  if ( *(_QWORD *)(v5 + 16) - (_QWORD)v6 <= 0xFu )
  {
    sub_16E7EE0(v5, "\"BRANCH-ON-MASK ", 0x10u);
    v7 = *(_QWORD *)(a1 + 40);
    if ( v7 )
      goto LABEL_5;
  }
  else
  {
    *v6 = _mm_load_si128((const __m128i *)&xmmword_42CA050);
    *(_QWORD *)(v5 + 24) += 16LL;
    v7 = *(_QWORD *)(a1 + 40);
    if ( v7 )
    {
LABEL_5:
      sub_1BE2750(a2, **(_QWORD **)(v7 + 40));
      goto LABEL_6;
    }
  }
  sub_1263B40(a2, " All-One");
LABEL_6:
  v8 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v8) <= 2 )
    return sub_16E7EE0(a2, "\\l\"", 3u);
  *(_BYTE *)(v8 + 2) = 34;
  *(_WORD *)v8 = 27740;
  *(_QWORD *)(a2 + 24) += 3LL;
  return 27740;
}

// Function: sub_1BE30D0
// Address: 0x1be30d0
//
__int64 __fastcall sub_1BE30D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // rdx
  void *v6; // rdx
  __int64 v7; // rdx

  v4 = a2;
  v5 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v5) <= 2 )
  {
    v4 = sub_16E7EE0(a2, " +\n", 3u);
  }
  else
  {
    *(_BYTE *)(v5 + 2) = 10;
    *(_WORD *)v5 = 11040;
    *(_QWORD *)(a2 + 24) += 3LL;
  }
  sub_16E2CE0(a3, v4);
  v6 = *(void **)(v4 + 24);
  if ( *(_QWORD *)(v4 + 16) - (_QWORD)v6 <= 0xAu )
  {
    v4 = sub_16E7EE0(v4, "\"WIDEN-PHI ", 0xBu);
  }
  else
  {
    qmemcpy(v6, "\"WIDEN-PHI ", 11);
    *(_QWORD *)(v4 + 24) += 11LL;
  }
  sub_1BE27E0(v4, *(_QWORD *)(a1 + 40));
  v7 = *(_QWORD *)(v4 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v4 + 16) - v7) <= 2 )
    return sub_16E7EE0(v4, "\\l\"", 3u);
  *(_BYTE *)(v7 + 2) = 34;
  *(_WORD *)v7 = 27740;
  *(_QWORD *)(v4 + 24) += 3LL;
  return 27740;
}

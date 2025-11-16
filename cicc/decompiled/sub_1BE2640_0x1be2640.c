// Function: sub_1BE2640
// Address: 0x1be2640
//
__int64 __fastcall sub_1BE2640(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rdx

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
  v6 = *(_QWORD *)(v5 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v5 + 16) - v6) <= 5 )
  {
    sub_16E7EE0(v5, "\"EMIT ", 6u);
  }
  else
  {
    *(_DWORD *)v6 = 1229800738;
    *(_WORD *)(v6 + 4) = 8276;
    *(_QWORD *)(v5 + 24) += 6LL;
  }
  sub_1BE2400(a1, a2);
  v7 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v7) <= 2 )
    return sub_16E7EE0(a2, "\\l\"", 3u);
  *(_BYTE *)(v7 + 2) = 34;
  *(_WORD *)v7 = 27740;
  *(_QWORD *)(a2 + 24) += 3LL;
  return 27740;
}

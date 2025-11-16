// Function: sub_E4E680
// Address: 0xe4e680
//
_BYTE *__fastcall sub_E4E680(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // rdi
  _WORD *v8; // rdx

  v5 = *(_QWORD *)(a1 + 304);
  v6 = *(_QWORD *)(v5 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v5 + 24) - v6) <= 6 )
  {
    sub_CB6200(v5, "\t.size\t", 7u);
  }
  else
  {
    *(_DWORD *)v6 = 1769156105;
    *(_WORD *)(v6 + 4) = 25978;
    *(_BYTE *)(v6 + 6) = 9;
    *(_QWORD *)(v5 + 32) += 7LL;
  }
  sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  v7 = *(_QWORD *)(a1 + 304);
  v8 = *(_WORD **)(v7 + 32);
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v8 <= 1u )
  {
    sub_CB6200(v7, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v8 = 8236;
    *(_QWORD *)(v7 + 32) += 2LL;
  }
  sub_E7FAD0(a3, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312), 0);
  return sub_E4D880(a1);
}

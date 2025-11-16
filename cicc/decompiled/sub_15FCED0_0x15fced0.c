// Function: sub_15FCED0
// Address: 0x15fced0
//
__int64 __fastcall sub_15FCED0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax

  sub_15F1F50(a1, a3, 42, a1 - 24, 1, a5);
  if ( *(_QWORD *)(a1 - 24) )
  {
    v6 = *(_QWORD *)(a1 - 16);
    v7 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v7 = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = *(_QWORD *)(v6 + 16) & 3LL | v7;
  }
  *(_QWORD *)(a1 - 24) = a2;
  if ( a2 )
  {
    v8 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a1 - 16) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = (a1 - 16) | *(_QWORD *)(v8 + 16) & 3LL;
    v9 = *(_QWORD *)(a1 - 8);
    *(_QWORD *)(a2 + 8) = a1 - 24;
    *(_QWORD *)(a1 - 8) = (a2 + 8) | v9 & 3;
  }
  return sub_164B780(a1, a4);
}

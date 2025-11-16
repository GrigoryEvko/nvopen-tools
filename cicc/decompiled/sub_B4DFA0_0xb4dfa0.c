// Function: sub_B4DFA0
// Address: 0xb4dfa0
//
__int64 __fastcall sub_B4DFA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v11; // rax
  __int64 v12; // rax
  bool v13; // zf
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax

  sub_B44260(a1, *(_QWORD *)(a2 + 8), 62, 3u, a7, a8);
  if ( *(_QWORD *)(a1 - 96) )
  {
    v11 = *(_QWORD *)(a1 - 88);
    **(_QWORD **)(a1 - 80) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = *(_QWORD *)(a1 - 80);
  }
  v12 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a1 - 96) = a2;
  *(_QWORD *)(a1 - 88) = v12;
  if ( v12 )
    *(_QWORD *)(v12 + 16) = a1 - 88;
  v13 = *(_QWORD *)(a1 - 64) == 0;
  *(_QWORD *)(a1 - 80) = a2 + 16;
  *(_QWORD *)(a2 + 16) = a1 - 96;
  if ( !v13 )
  {
    v14 = *(_QWORD *)(a1 - 56);
    **(_QWORD **)(a1 - 48) = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(a1 - 48);
  }
  *(_QWORD *)(a1 - 64) = a3;
  if ( a3 )
  {
    v15 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a1 - 56) = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = a1 - 56;
    *(_QWORD *)(a1 - 48) = a3 + 16;
    *(_QWORD *)(a3 + 16) = a1 - 64;
  }
  if ( *(_QWORD *)(a1 - 32) )
  {
    v16 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a4;
  if ( a4 )
  {
    v17 = *(_QWORD *)(a4 + 16);
    *(_QWORD *)(a1 - 24) = v17;
    if ( v17 )
      *(_QWORD *)(v17 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a4 + 16;
    *(_QWORD *)(a4 + 16) = a1 - 32;
  }
  return sub_BD6B50(a1, a5);
}

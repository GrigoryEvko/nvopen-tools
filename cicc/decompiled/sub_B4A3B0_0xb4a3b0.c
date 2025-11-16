// Function: sub_B4A3B0
// Address: 0xb4a3b0
//
__int64 __fastcall sub_B4A3B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool v4; // zf
  __int64 v5; // rax
  __int64 v6; // rax

  v4 = *(_QWORD *)(a1 - 32) == 0;
  *(_QWORD *)(a1 + 80) = a2;
  if ( !v4 )
  {
    v5 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a3;
  if ( a3 )
  {
    v6 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a1 - 24) = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = a1 - 32;
  }
  return sub_BD6B50(a1, a4);
}

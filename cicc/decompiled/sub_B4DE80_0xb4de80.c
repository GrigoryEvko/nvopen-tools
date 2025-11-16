// Function: sub_B4DE80
// Address: 0xb4de80
//
__int64 __fastcall sub_B4DE80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int16 a6)
{
  __int64 v8; // rax
  __int64 v9; // rax
  bool v10; // zf
  __int64 v11; // rax
  __int64 v12; // rax

  sub_B44260(a1, *(_QWORD *)(*(_QWORD *)(a2 + 8) + 24LL), 61, 2u, a5, a6);
  if ( *(_QWORD *)(a1 - 64) )
  {
    v8 = *(_QWORD *)(a1 - 56);
    **(_QWORD **)(a1 - 48) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = *(_QWORD *)(a1 - 48);
  }
  v9 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a1 - 64) = a2;
  *(_QWORD *)(a1 - 56) = v9;
  if ( v9 )
    *(_QWORD *)(v9 + 16) = a1 - 56;
  v10 = *(_QWORD *)(a1 - 32) == 0;
  *(_QWORD *)(a1 - 48) = a2 + 16;
  *(_QWORD *)(a2 + 16) = a1 - 64;
  if ( !v10 )
  {
    v11 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a3;
  if ( a3 )
  {
    v12 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a1 - 24) = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = a1 - 32;
  }
  return sub_BD6B50(a1, a4);
}

// Function: sub_B51A10
// Address: 0xb51a10
//
__int64 __fastcall sub_B51A10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int16 a6)
{
  __int64 v7; // rax
  __int64 v8; // rax

  sub_B44260(a1, a3, 42, 1u, a5, a6);
  if ( *(_QWORD *)(a1 - 32) )
  {
    v7 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v7;
    if ( v7 )
      *(_QWORD *)(v7 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a2;
  if ( a2 )
  {
    v8 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(a1 - 24) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = a1 - 32;
  }
  return sub_BD6B50(a1, a4);
}

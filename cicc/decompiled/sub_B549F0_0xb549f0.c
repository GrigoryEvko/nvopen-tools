// Function: sub_B549F0
// Address: 0xb549f0
//
__int64 __fastcall sub_B549F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int16 a5)
{
  __int64 v6; // rax
  __int64 v7; // rax

  sub_B44260(a1, *(_QWORD *)(a2 + 8), 67, 1u, a4, a5);
  if ( *(_QWORD *)(a1 - 32) )
  {
    v6 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = *(_QWORD *)(a1 - 16);
  }
  v7 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a1 - 32) = a2;
  *(_QWORD *)(a1 - 24) = v7;
  if ( v7 )
    *(_QWORD *)(v7 + 16) = a1 - 24;
  *(_QWORD *)(a1 - 16) = a2 + 16;
  *(_QWORD *)(a2 + 16) = a1 - 32;
  return sub_BD6B50(a1, a3);
}

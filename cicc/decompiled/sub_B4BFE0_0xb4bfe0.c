// Function: sub_B4BFE0
// Address: 0xb4bfe0
//
void __fastcall sub_B4BFE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax

  if ( *(_QWORD *)(a1 - 64) )
  {
    v3 = *(_QWORD *)(a1 - 56);
    **(_QWORD **)(a1 - 48) = v3;
    if ( v3 )
      *(_QWORD *)(v3 + 16) = *(_QWORD *)(a1 - 48);
  }
  *(_QWORD *)(a1 - 64) = a2;
  if ( a2 )
  {
    v4 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(a1 - 56) = v4;
    if ( v4 )
      *(_QWORD *)(v4 + 16) = a1 - 56;
    *(_QWORD *)(a1 - 48) = a2 + 16;
    *(_QWORD *)(a2 + 16) = a1 - 64;
  }
  if ( *(_QWORD *)(a1 - 32) )
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
}

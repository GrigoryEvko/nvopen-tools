// Function: sub_AC2B30
// Address: 0xac2b30
//
void __fastcall sub_AC2B30(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax

  if ( *(_QWORD *)a1 )
  {
    v2 = *(_QWORD *)(a1 + 8);
    **(_QWORD **)(a1 + 16) = v2;
    if ( v2 )
      *(_QWORD *)(v2 + 16) = *(_QWORD *)(a1 + 16);
  }
  *(_QWORD *)a1 = a2;
  if ( a2 )
  {
    v3 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(a1 + 8) = v3;
    if ( v3 )
      *(_QWORD *)(v3 + 16) = a1 + 8;
    *(_QWORD *)(a1 + 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = a1;
  }
}

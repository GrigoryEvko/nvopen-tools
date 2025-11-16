// Function: sub_B303B0
// Address: 0xb303b0
//
void __fastcall sub_B303B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax

  if ( *(_QWORD *)(a1 - 32) )
  {
    v2 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v2;
    if ( v2 )
      *(_QWORD *)(v2 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a2;
  if ( a2 )
  {
    v3 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(a1 - 24) = v3;
    if ( v3 )
      *(_QWORD *)(v3 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = a1 - 32;
  }
}

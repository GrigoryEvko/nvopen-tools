// Function: sub_12D3BB0
// Address: 0x12d3bb0
//
__int64 __fastcall sub_12D3BB0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 32);
  if ( *(_QWORD *)(a1 + 40) == v1 )
    return 0;
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(v1 + 8);
  return 1;
}

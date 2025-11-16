// Function: sub_12D3B80
// Address: 0x12d3b80
//
__int64 __fastcall sub_12D3B80(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 48);
  if ( *(_QWORD *)(a1 + 56) == v1 )
    return 0;
  *(_QWORD *)(a1 + 48) = *(_QWORD *)(v1 + 8);
  return 1;
}

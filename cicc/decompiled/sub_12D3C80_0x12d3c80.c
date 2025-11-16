// Function: sub_12D3C80
// Address: 0x12d3c80
//
__int64 __fastcall sub_12D3C80(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 16);
  if ( *(_QWORD *)(a1 + 24) == v1 || !v1 )
    return 0;
  else
    return v1 - 48;
}

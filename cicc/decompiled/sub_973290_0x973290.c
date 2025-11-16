// Function: sub_973290
// Address: 0x973290
//
__int64 __fastcall sub_973290(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rax
  char v3; // r8

  if ( !a1 )
    return 0;
  if ( *(_BYTE *)a1 != 85 )
    return 0;
  v2 = *(_QWORD *)(a1 - 32);
  if ( !v2 )
    return 0;
  if ( *(_BYTE *)v2 )
    return 0;
  if ( *(_QWORD *)(v2 + 24) != *(_QWORD *)(a1 + 80) )
    return 0;
  if ( (*(_BYTE *)(v2 + 33) & 0x20) == 0 )
    return 0;
  v3 = sub_B5A1B0(a1);
  result = a1;
  if ( !v3 )
    return 0;
  return result;
}

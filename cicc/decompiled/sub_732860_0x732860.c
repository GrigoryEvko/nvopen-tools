// Function: sub_732860
// Address: 0x732860
//
__int64 __fastcall sub_732860(__int64 a1)
{
  __int64 i; // rax
  __int64 result; // rax

  for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  result = **(_QWORD **)(i + 168);
  if ( !result )
    return 0;
  while ( (*(_BYTE *)(result + 32) & 4) == 0 )
  {
    result = *(_QWORD *)result;
    if ( !result )
      return result;
  }
  return 1;
}

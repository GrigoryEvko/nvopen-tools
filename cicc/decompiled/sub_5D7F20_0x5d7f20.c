// Function: sub_5D7F20
// Address: 0x5d7f20
//
_BOOL8 __fastcall sub_5D7F20(__int64 a1)
{
  __int64 i; // rax

  if ( *(_QWORD *)(a1 + 8) )
    return 0;
  if ( !(unsigned int)sub_8D3A70(*(_QWORD *)(a1 + 120)) )
    return 0;
  for ( i = *(_QWORD *)(a1 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  return (*(_BYTE *)(i + 177) & 4) != 0;
}

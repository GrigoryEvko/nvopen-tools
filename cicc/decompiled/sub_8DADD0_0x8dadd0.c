// Function: sub_8DADD0
// Address: 0x8dadd0
//
_BOOL8 __fastcall sub_8DADD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  char i; // al

  if ( !(dword_4F06978 | dword_4D048B8) )
    return 0;
  while ( 1 )
  {
    v6 = *(unsigned __int8 *)(a1 + 140);
    if ( (_BYTE)v6 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  for ( i = *(_BYTE *)(a2 + 140); i == 12; i = *(_BYTE *)(a2 + 140) )
    a2 = *(_QWORD *)(a2 + 160);
  return i
      && (_BYTE)v6
      && sub_8DAC40(*(_QWORD *)(*(_QWORD *)(a1 + 168) + 56LL), *(_QWORD *)(*(_QWORD *)(a2 + 168) + 56LL), v6, a4, a5);
}

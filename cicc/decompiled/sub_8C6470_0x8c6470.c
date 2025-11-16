// Function: sub_8C6470
// Address: 0x8c6470
//
__int64 __fastcall sub_8C6470(__int64 a1)
{
  __int64 v1; // r12
  char i; // al
  __int64 result; // rax

  v1 = a1;
  for ( i = *(_BYTE *)(a1 + 140); i == 12; i = *(_BYTE *)(v1 + 140) )
    v1 = *(_QWORD *)(v1 + 160);
  if ( (unsigned __int8)(i - 9) <= 2u )
    return sub_8D2490(v1);
  if ( !(unsigned int)sub_8D28B0(v1) && (!(unsigned int)sub_8D2870(v1) || (*(_BYTE *)(v1 + 161) & 4) == 0) )
    return (unsigned int)sub_8D23B0(v1) == 0;
  result = 0;
  if ( (*(_BYTE *)(v1 + 141) & 0x20) == 0 )
    return **(_BYTE **)(v1 + 176) & 1;
  return result;
}

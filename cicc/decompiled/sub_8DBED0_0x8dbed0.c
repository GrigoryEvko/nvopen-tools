// Function: sub_8DBED0
// Address: 0x8dbed0
//
__int64 __fastcall sub_8DBED0(__int64 a1)
{
  __int64 i; // r12

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
  {
    if ( (*(_BYTE *)(i + 186) & 8) != 0 && (unsigned int)sub_8DBE70(*(_QWORD *)(i + 160)) )
      break;
  }
  return i;
}

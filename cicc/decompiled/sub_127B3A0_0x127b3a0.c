// Function: sub_127B3A0
// Address: 0x127b3a0
//
__int64 __fastcall sub_127B3A0(__int64 a1)
{
  char v1; // al
  unsigned int v2; // r8d

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  v2 = 0;
  if ( v1 == 2 )
    LOBYTE(v2) = byte_4B6DF90[*(unsigned __int8 *)(a1 + 160)] != 0;
  return v2;
}

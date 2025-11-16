// Function: sub_8D27E0
// Address: 0x8d27e0
//
__int64 __fastcall sub_8D27E0(__int64 a1)
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
    return byte_4B6DF90[*(unsigned __int8 *)(a1 + 160)] != 0;
  return v2;
}

// Function: sub_8D2820
// Address: 0x8d2820
//
_BOOL8 __fastcall sub_8D2820(__int64 a1)
{
  char v1; // al

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  return v1 == 2 && (*(_DWORD *)(a1 + 160) & 0x7C800) == 0 && (unsigned int)*(unsigned __int8 *)(a1 + 160) - 1 <= 9;
}

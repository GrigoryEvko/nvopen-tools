// Function: sub_8D3A00
// Address: 0x8d3a00
//
_BOOL8 __fastcall sub_8D3A00(__int64 a1)
{
  char v1; // al

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  return v1 == 2 && (unk_4D04000 || (*(_BYTE *)(a1 + 161) & 8) == 0) && *(_BYTE *)(a1 + 160) == byte_4F06A51[0];
}

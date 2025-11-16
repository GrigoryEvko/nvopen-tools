// Function: sub_8D29E0
// Address: 0x8d29e0
//
_BOOL8 __fastcall sub_8D29E0(__int64 a1)
{
  char v1; // al

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  return v1 == 2
      && (unk_4D04000 || (*(_BYTE *)(a1 + 161) & 8) == 0)
      && *(_BYTE *)(a1 + 160) <= 2u
      && (*(_DWORD *)(a1 + 160) & 0x7C800) == 0;
}

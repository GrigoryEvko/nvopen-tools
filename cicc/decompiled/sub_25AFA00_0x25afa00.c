// Function: sub_25AFA00
// Address: 0x25afa00
//
__int64 __fastcall sub_25AFA00(__int64 a1, __int64 a2)
{
  if ( sub_B2FC80(a2) || (*(_BYTE *)(a2 + 32) & 0xFu) - 7 > 1 )
    return 0;
  else
    return sub_25AEA80(a2);
}

// Function: sub_2C1A110
// Address: 0x2c1a110
//
__int64 __fastcall sub_2C1A110(__int64 a1)
{
  return (((*(_BYTE *)(a1 + 156) & 0x40) != 0) << 6)
       | (32 * ((*(_BYTE *)(a1 + 156) & 0x20) != 0))
       | (16 * ((*(_BYTE *)(a1 + 156) & 0x10) != 0))
       | (8 * ((*(_BYTE *)(a1 + 156) & 8) != 0))
       | *(_BYTE *)(a1 + 156) & 1
       | (2 * ((*(_BYTE *)(a1 + 156) & 2) != 0))
       | (4 * (unsigned int)((*(_BYTE *)(a1 + 156) & 4) != 0));
}

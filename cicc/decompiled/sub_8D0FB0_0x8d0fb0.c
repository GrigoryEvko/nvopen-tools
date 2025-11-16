// Function: sub_8D0FB0
// Address: 0x8d0fb0
//
__int64 __fastcall sub_8D0FB0(__int64 a1)
{
  unsigned __int8 v1; // al
  char v2; // al

  v1 = *(_BYTE *)(a1 + 140);
  if ( v1 > 0xBu )
  {
    if ( v1 != 15 )
      return 0;
    byte_4F60594 |= 2u;
    return 0;
  }
  else
  {
    if ( v1 <= 8u )
    {
      if ( v1 == 2 )
      {
        if ( (unsigned __int8)(*(_BYTE *)(a1 + 160) - 11) <= 1u && unk_4D04710 )
          byte_4F60594 |= 1u;
      }
      else if ( v1 == 3 )
      {
        v2 = *(_BYTE *)(a1 + 160);
        if ( v2 == 8 || v2 == 13 )
        {
          if ( unk_4D0470C )
            byte_4F60594 |= 4u;
        }
      }
      return 0;
    }
    byte_4F60594 |= (*(_WORD *)(a1 + 180) >> 6) & 0xF;
    return 0;
  }
}

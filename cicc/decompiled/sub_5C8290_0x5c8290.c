// Function: sub_5C8290
// Address: 0x5c8290
//
__int64 __fastcall sub_5C8290(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rdi

  if ( *(_BYTE *)(a2 + 40) != 24 )
  {
    v3 = a1 + 56;
    v4 = 8;
    if ( !unk_4F077B4 )
      v4 = unk_4F07471;
    sub_684AA0(v4, 2812, v3);
    *(_BYTE *)(a1 + 8) = 0;
    return a2;
  }
  if ( !(unsigned int)sub_86F930() )
  {
    sub_6851C0(2813, a1 + 56);
    *(_BYTE *)(a1 + 8) = 0;
    return a2;
  }
  *(_BYTE *)(a2 + 41) |= 8u;
  return a2;
}

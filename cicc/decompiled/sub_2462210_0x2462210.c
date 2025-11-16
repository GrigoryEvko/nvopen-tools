// Function: sub_2462210
// Address: 0x2462210
//
__int64 __fastcall sub_2462210(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rdx
  __int64 v5; // rdx
  __int64 v6; // [rsp+0h] [rbp-20h] BYREF
  __int64 v7; // [rsp+8h] [rbp-18h]

  v2 = *(_BYTE *)(a2 + 8);
  if ( (v2 & 0xFD) == 0xC )
  {
    v6 = sub_BCAE30(a2);
    v7 = v3;
    if ( (unsigned __int64)sub_CA1930(&v6) <= 0x40 )
      return 0;
    v2 = *(_BYTE *)(a2 + 8);
  }
  if ( v2 <= 3u || v2 == 5 || (v2 & 0xFD) == 4 )
  {
    v6 = sub_BCAE30(a2);
    v7 = v5;
    if ( (unsigned __int64)sub_CA1930(&v6) <= 0x80 )
      return 1;
    v2 = *(_BYTE *)(a2 + 8);
  }
  if ( v2 == 16 || v2 == 17 )
    return sub_2462210(a1, **(_QWORD **)(a2 + 16));
  else
    return 2;
}

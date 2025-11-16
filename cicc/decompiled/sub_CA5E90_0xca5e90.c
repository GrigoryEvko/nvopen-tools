// Function: sub_CA5E90
// Address: 0xca5e90
//
__int64 __fastcall sub_CA5E90(char *a1, __int64 a2)
{
  unsigned __int64 v2; // rsi
  int v4; // eax

  v2 = (unsigned __int64)&a1[a2];
  if ( v2 > (unsigned __int64)a1 && *a1 >= 0 )
    return (unsigned int)*a1 | 0x100000000LL;
  if ( v2 > (unsigned __int64)(a1 + 1)
    && (*a1 & 0xE0) == 0xC0
    && (a1[1] & 0xC0) == 0x80
    && (a1[1] & 0x3F | (*a1 << 6) & 0x7C0) > 127 )
  {
    return a1[1] & 0x3F | (*a1 << 6) & 0x7C0u | 0x200000000LL;
  }
  if ( v2 > (unsigned __int64)(a1 + 2) && (*a1 & 0xF0) == 0xE0 && (a1[1] & 0xC0) == 0x80 && (a1[2] & 0xC0) == 0x80 )
  {
    v4 = a1[2] & 0x3F | (a1[1] << 6) & 0xFC0 | (unsigned __int16)(*a1 << 12);
    if ( v4 > 2047 && (unsigned int)(v4 - 55296) > 0x7FF )
      return (unsigned __int16)v4 | 0x300000000LL;
  }
  if ( v2 > (unsigned __int64)(a1 + 3)
    && (*a1 & 0xF8) == 0xF0
    && (a1[1] & 0xC0) == 0x80
    && (a1[2] & 0xC0) == 0x80
    && (a1[3] & 0xC0) == 0x80
    && ((a1[2] << 6) & 0xFC0 | a1[3] & 0x3F | (a1[1] << 12) & 0x3F000 | (*a1 << 18) & 0x1C0000u) - 0x10000 <= 0xFFFFF )
  {
    return (a1[2] << 6) & 0xFC0 | a1[3] & 0x3F | (a1[1] << 12) & 0x3F000 | (*a1 << 18) & 0x1C0000u | 0x400000000LL;
  }
  return 0;
}

// Function: sub_16F61C0
// Address: 0x16f61c0
//
__int64 __fastcall sub_16F61C0(char *a1, __int64 a2)
{
  char v2; // al
  char *v4; // rsi
  int v5; // edx
  int v6; // ecx
  __int16 v7; // si
  char v8; // di
  int v9; // eax

  v2 = *a1;
  if ( *a1 >= 0 )
    return (unsigned int)v2 | 0x100000000LL;
  v4 = &a1[a2];
  if ( v4 != a1 + 1 && (v2 & 0xE0) == 0xC0 && (a1[1] & 0xC0) == 0x80 && (a1[1] & 0x3F | (v2 << 6) & 0x7C0) > 127 )
    return a1[1] & 0x3F | (v2 << 6) & 0x7C0u | 0x200000000LL;
  if ( v4 != a1 + 2 && (v2 & 0xF0) == 0xE0 && (a1[1] & 0xC0) == 0x80 && (a1[2] & 0xC0) == 0x80 )
  {
    v5 = a1[2] & 0x3F | (a1[1] << 6) & 0xFC0 | (unsigned __int16)(v2 << 12);
    if ( v5 > 2047 && (unsigned int)(v5 - 55296) > 0x7FF )
      return (unsigned __int16)v5 | 0x300000000LL;
  }
  if ( v4 != a1 + 3 && (v2 & 0xF8) == 0xF0 )
  {
    v6 = a1[1];
    if ( (a1[1] & 0xC0) == 0x80 )
    {
      v7 = a1[2];
      if ( (a1[2] & 0xC0) == 0x80 )
      {
        v8 = a1[3];
        if ( (v8 & 0xC0) == 0x80 )
        {
          v9 = (v7 << 6) & 0xFC0 | v8 & 0x3F | (v6 << 12) & 0x3F000 | (v2 << 18) & 0x1C0000;
          if ( (unsigned int)(v9 - 0x10000) <= 0xFFFFF )
            return v9 & 0x1FFFFF | 0x400000000LL;
        }
      }
    }
  }
  return 0;
}

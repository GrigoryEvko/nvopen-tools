// Function: sub_15B16B0
// Address: 0x15b16b0
//
__int64 __fastcall sub_15B16B0(unsigned int a1, __int64 a2)
{
  unsigned int v2; // r12d
  int v3; // ebx
  int v4; // ebx
  _DWORD v6[9]; // [rsp+Ch] [rbp-24h] BYREF

  v2 = a1;
  v3 = a1 & 3;
  if ( (a1 & 3) != 0 )
  {
    if ( v3 == 1 )
    {
      v6[0] = 1;
      sub_15B1660(a2, v6);
    }
    else
    {
      if ( v3 == 2 )
        v6[0] = 2;
      else
        v6[0] = 3;
      sub_15B1660(a2, v6);
    }
    v2 = (v3 ^ 0x7FFFFFF) & a1;
  }
  v4 = v2 & 0x30000;
  if ( (v2 & 0x30000) != 0 )
  {
    if ( v4 == 0x10000 )
    {
      v6[0] = 0x10000;
      sub_15B1660(a2, v6);
    }
    else
    {
      if ( v4 == 0x20000 )
        v6[0] = 0x20000;
      else
        v6[0] = 196608;
      sub_15B1660(a2, v6);
    }
    v2 &= v4 ^ 0x7FFFFFF;
  }
  if ( (v2 & 0x24) == 0x24 )
  {
    v6[0] = 36;
    v2 &= 0x7FFFFDBu;
    sub_15B1660(a2, v6);
  }
  v6[0] = v2 & 1;
  if ( (v2 & 1) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 2;
  if ( (v2 & 2) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 3;
  if ( (v2 & 3) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 4;
  if ( (v2 & 4) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 8;
  if ( (v2 & 8) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x10;
  if ( (v2 & 0x10) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x20;
  if ( (v2 & 0x20) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x40;
  if ( (v2 & 0x40) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x80;
  if ( (v2 & 0x80) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x100;
  if ( (v2 & 0x100) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x200;
  if ( (v2 & 0x200) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x400;
  if ( (v2 & 0x400) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x800;
  if ( (v2 & 0x800) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x1000;
  if ( (v2 & 0x1000) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x2000;
  if ( (v2 & 0x2000) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x4000;
  if ( (v2 & 0x4000) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x8000;
  if ( (v2 & 0x8000) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x10000;
  if ( (v2 & 0x10000) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x20000;
  if ( (v2 & 0x20000) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x30000;
  if ( (v2 & 0x30000) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x40000;
  if ( (v2 & 0x40000) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x80000;
  if ( (v2 & 0x80000) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x100000;
  if ( (v2 & 0x100000) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x200000;
  if ( (v2 & 0x200000) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x400000;
  if ( (v2 & 0x400000) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x800000;
  if ( (v2 & 0x800000) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x1000000;
  if ( (v2 & 0x1000000) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x2000000;
  if ( (v2 & 0x2000000) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x4000000;
  if ( (v2 & 0x4000000) != 0 )
  {
    sub_15B1660(a2, v6);
    v2 &= ~v6[0] & 0x7FFFFFF;
  }
  v6[0] = v2 & 0x24;
  if ( (v2 & 0x24) == 0 )
    return v2;
  sub_15B1660(a2, v6);
  return ~v6[0] & v2 & 0x7FFFFFF;
}

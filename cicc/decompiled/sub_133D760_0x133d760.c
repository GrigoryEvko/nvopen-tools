// Function: sub_133D760
// Address: 0x133d760
//
__int64 __fastcall sub_133D760(__int64 *a1)
{
  __int64 result; // rax
  unsigned __int64 v2; // rax
  __int64 v3; // rsi
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rcx
  char v7; // cl
  unsigned __int64 v8; // rax
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_130B140(a1 + 19, a1 + 17);
  sub_130B1D0(a1 + 19, a1 + 16);
  result = a1[15];
  if ( result > 0 )
  {
    v2 = sub_130B0E0((__int64)(a1 + 16));
    v3 = 0;
    v4 = v2;
    if ( v2 != 1 )
    {
      if ( v2 <= 1 )
      {
        v7 = 65;
      }
      else
      {
        _BitScanReverse64(&v5, v2 - 1);
        if ( !_BitScanForward64(&v2, 1LL << ((unsigned __int8)v5 + 1)) )
          LOBYTE(v2) = -1;
        v7 = 64 - v2;
      }
      v8 = a1[18];
      do
      {
        v8 = 0x5851F42D4C957F2DLL * v8 + 0x14057B7EF767814FLL;
        v3 = v8 >> v7;
      }
      while ( v4 <= v8 >> v7 );
      a1[18] = v8;
    }
    sub_130B0C0(v9, v3);
    return sub_130B1D0(a1 + 19, v9);
  }
  return result;
}

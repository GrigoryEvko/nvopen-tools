// Function: sub_12FC3F0
// Address: 0x12fc3f0
//
__int64 *__fastcall sub_12FC3F0(__int64 *a1, unsigned __int64 a2, unsigned __int64 a3)
{
  __int64 *result; // rax
  unsigned __int64 v4; // rcx
  char v5; // r8
  char v6; // cl
  __int64 v7; // rdi
  unsigned __int64 v8; // r8

  result = a1;
  if ( a2 )
  {
    _BitScanReverse64(&v8, a2);
    LOBYTE(v8) = (v8 ^ 0x3F) - 15;
    *a1 = 1 - (char)v8;
    a1[1] = a3 << v8;
    a1[2] = (a2 << v8) | (a3 >> -(char)v8);
  }
  else
  {
    if ( a3 )
    {
      _BitScanReverse64(&v4, a3);
      v5 = (v4 ^ 0x3F) - 15;
      v6 = v5;
      v7 = -63 - v5;
      if ( v5 < 0 )
      {
        *result = v7;
        result[2] = a3 >> -v5;
        result[1] = a3 << v5;
        return result;
      }
    }
    else
    {
      v7 = -112;
      v6 = 49;
    }
    *result = v7;
    result[1] = 0;
    result[2] = a3 << v6;
  }
  return result;
}

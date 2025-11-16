// Function: sub_C1B1E0
// Address: 0xc1b1e0
//
__int64 __fastcall sub_C1B1E0(unsigned __int64 a1, unsigned __int64 a2, unsigned __int64 a3, bool *a4)
{
  unsigned __int64 v4; // r8
  int v5; // r9d
  unsigned __int64 v6; // r8
  unsigned int v7; // eax
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // r8
  char v13; // [rsp+1h] [rbp-1h] BYREF

  if ( !a4 )
    a4 = (bool *)&v13;
  if ( a1
    && (_BitScanReverse64(&v4, a1), v5 = 63 - (v4 ^ 0x3F), a2)
    && (_BitScanReverse64(&v6, a2), v7 = v5 + 63 - (v6 ^ 0x3F), v7 > 0x3E) )
  {
    if ( v7 != 63 || (v8 = a2 * (a1 >> 1), v8 < 0) )
    {
      *a4 = 1;
      return -1;
    }
    v9 = 2 * v8;
    if ( (a1 & 1) != 0 )
    {
      v10 = a2 + v9;
      if ( a2 < v9 )
        a2 = v9;
      *a4 = v10 < a2;
      if ( v10 < a2 )
        return -1;
      v9 = v10;
    }
  }
  else
  {
    v9 = a2 * a1;
  }
  v11 = a3 + v9;
  if ( a3 < v9 )
    a3 = v9;
  *a4 = v11 < a3;
  if ( v11 >= a3 )
    return v11;
  return -1;
}

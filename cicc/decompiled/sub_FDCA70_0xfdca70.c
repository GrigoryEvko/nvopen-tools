// Function: sub_FDCA70
// Address: 0xfdca70
//
__int64 __fastcall sub_FDCA70(unsigned __int64 *a1, unsigned __int16 *a2, unsigned __int64 *a3, unsigned __int16 *a4)
{
  __int64 result; // rax
  unsigned int v6; // r8d
  unsigned __int16 *v7; // rdx
  __int16 v8; // cx
  unsigned __int16 *v9; // rcx
  unsigned __int64 *v10; // rcx
  unsigned __int64 v11; // r10
  int v12; // eax
  unsigned __int64 v13; // rcx
  int v14; // ecx
  char v15; // r8
  __int16 v16; // r11

  LODWORD(result) = (__int16)*a2;
  v6 = (__int16)*a4;
  v7 = a4;
  while ( (__int16)result < (__int16)v6 )
  {
    v8 = result;
    LODWORD(result) = (__int16)v6;
    v6 = v8;
    v9 = a2;
    a2 = v7;
    v7 = v9;
    v10 = a1;
    a1 = a3;
    a3 = v10;
  }
  v11 = *a1;
  if ( !*a1 )
    return v6;
  if ( !*a3 || (_WORD)result == (_WORD)v6 )
    return (unsigned int)result;
  v12 = result - v6;
  if ( v12 <= 127 )
  {
    _BitScanReverse64(&v13, v11);
    v14 = v13 ^ 0x3F;
    if ( v12 < v14 )
    {
      v16 = 0;
      v15 = 0;
      goto LABEL_11;
    }
    v12 -= v14;
    v15 = v12;
    if ( v12 <= 63 )
    {
      v16 = v12;
      LOWORD(v12) = v14;
LABEL_11:
      *a1 = v11 << v12;
      *a3 >>= v15;
      *a2 -= v12;
      *v7 += v16;
      return *a2;
    }
  }
  *a3 = 0;
  return *a2;
}

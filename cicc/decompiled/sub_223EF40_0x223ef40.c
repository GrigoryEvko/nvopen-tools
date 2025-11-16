// Function: sub_223EF40
// Address: 0x223ef40
//
__int64 __fastcall sub_223EF40(void *dest, unsigned __int64 a2, unsigned __int64 a3)
{
  _BYTE *v5; // rsi
  unsigned __int64 v6; // rax
  _BYTE v9[24]; // [rsp+18h] [rbp-18h] BYREF

  v5 = v9;
  do
  {
    *--v5 = a0123456789[a3 % 0xA + 1];
    v6 = a3;
    a3 /= 0xAu;
  }
  while ( v6 > 9 );
  if ( v9 - v5 > a2 )
    return 0xFFFFFFFFLL;
  memcpy(dest, v9 - 8 * (v9 - v5) + 32, v9 - v5);
  return (unsigned int)(v9 - v5);
}

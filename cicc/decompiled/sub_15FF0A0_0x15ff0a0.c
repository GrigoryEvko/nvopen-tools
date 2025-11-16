// Function: sub_15FF0A0
// Address: 0x15ff0a0
//
__int64 __fastcall sub_15FF0A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  int v5; // eax

  v5 = *(unsigned __int16 *)(a1 + 18);
  BYTE1(v5) &= ~0x80u;
  if ( *(_BYTE *)(a1 + 16) == 75 )
  {
    LOBYTE(a5) = (unsigned int)(v5 - 32) <= 1;
    return a5;
  }
  LOBYTE(a5) = v5 == 6 || v5 == 1;
  if ( (_BYTE)a5 )
    return a5;
  LOBYTE(a5) = v5 == 9;
  LOBYTE(v5) = v5 == 14;
  return v5 | a5;
}

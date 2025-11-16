// Function: sub_24F52B0
// Address: 0x24f52b0
//
unsigned __int64 __fastcall sub_24F52B0(unsigned __int8 *a1)
{
  int v1; // ecx
  unsigned __int64 result; // rax
  unsigned __int8 v3; // cl

  v1 = *a1;
  result = (unsigned int)(v1 - 67);
  LOBYTE(result) = (_BYTE)v1 == 63 || (unsigned int)result <= 0xC;
  if ( !(_BYTE)result )
  {
    v3 = v1 - 42;
    if ( v3 <= 0x2Cu )
      return (0x13000003FFFFuLL >> v3) & 1;
  }
  return result;
}

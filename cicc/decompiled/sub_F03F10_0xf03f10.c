// Function: sub_F03F10
// Address: 0xf03f10
//
__int64 __fastcall sub_F03F10(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rcx
  unsigned __int8 v3; // si
  unsigned __int8 v4; // dl

  LODWORD(result) = 0;
  v2 = a1 >> 63;
  v3 = a1 >> 63;
  do
  {
    do
    {
      v4 = a1;
      a1 >>= 7;
      result = (unsigned int)(result + 1);
    }
    while ( v2 != a1 );
  }
  while ( ((v3 ^ v4) & 0x40) != 0 );
  return result;
}

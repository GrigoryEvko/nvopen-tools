// Function: sub_D00090
// Address: 0xd00090
//
__int64 __fastcall sub_D00090(unsigned int a1, int a2)
{
  __int64 result; // rax

  if ( (_BYTE)a1 != (_BYTE)a2 || (result = a1, ((BYTE1(a2) ^ BYTE1(a1)) & 1) != 0) || ((a2 ^ a1) & 0xFFFFFE00) != 0 )
  {
    if ( (_BYTE)a1 == 2 && (_BYTE)a2 == 3 || (_BYTE)a2 == 2 && (_BYTE)a1 == 3 )
      return 2;
    else
      return 1;
  }
  return result;
}

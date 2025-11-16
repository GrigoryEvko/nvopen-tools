// Function: sub_87D600
// Address: 0x87d600
//
__int64 __fastcall sub_87D600(unsigned __int8 a1, unsigned int a2)
{
  __int64 result; // rax

  if ( (_BYTE)a2 == 3 || a1 > 1u )
    return 3;
  result = 2;
  if ( (_BYTE)a2 != 2 )
  {
    result = 1;
    if ( (_BYTE)a2 == a1 )
      return a2;
  }
  return result;
}

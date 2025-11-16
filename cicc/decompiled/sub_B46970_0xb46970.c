// Function: sub_B46970
// Address: 0xb46970
//
__int64 __fastcall sub_B46970(unsigned __int8 *a1)
{
  __int64 result; // rax

  result = sub_B46490((__int64)a1);
  if ( !(_BYTE)result )
  {
    result = sub_B46790(a1, 0);
    if ( !(_BYTE)result )
      return (unsigned int)sub_B46900(a1) ^ 1;
  }
  return result;
}

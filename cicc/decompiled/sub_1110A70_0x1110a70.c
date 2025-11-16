// Function: sub_1110A70
// Address: 0x1110a70
//
__int64 __fastcall sub_1110A70(_BYTE *a1)
{
  __int64 result; // rax

  result = 1;
  if ( *a1 <= 0x15u )
  {
    result = 0;
    if ( *a1 != 5 )
      return (unsigned int)sub_AD6CA0((__int64)a1) ^ 1;
  }
  return result;
}

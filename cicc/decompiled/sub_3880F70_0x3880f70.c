// Function: sub_3880F70
// Address: 0x3880f70
//
__int64 __fastcall sub_3880F70(unsigned __int8 **a1)
{
  __int64 result; // rax

  do
  {
    result = **a1;
    if ( (_BYTE)result == 10 )
      break;
    if ( (_BYTE)result == 13 )
      break;
    result = sub_3880F40(a1);
  }
  while ( (_DWORD)result != -1 );
  return result;
}

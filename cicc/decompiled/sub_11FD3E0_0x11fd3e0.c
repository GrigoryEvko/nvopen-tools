// Function: sub_11FD3E0
// Address: 0x11fd3e0
//
__int64 __fastcall sub_11FD3E0(unsigned __int8 **a1)
{
  __int64 result; // rax

  do
  {
    result = **a1;
    if ( (_BYTE)result == 10 )
      break;
    if ( (_BYTE)result == 13 )
      break;
    result = sub_11FD3B0(a1);
  }
  while ( (_DWORD)result != -1 );
  return result;
}

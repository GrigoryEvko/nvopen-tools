// Function: sub_7AF220
// Address: 0x7af220
//
__int64 __fastcall sub_7AF220(unsigned __int64 a1)
{
  __int64 result; // rax

  result = qword_4F08580[(a1 >> 3) - 7993 * (a1 / 0xF9C8)];
  if ( !result )
    return 0;
  while ( *(_QWORD *)(result + 16) != a1 )
  {
    result = *(_QWORD *)(result + 8);
    if ( !result )
      return result;
  }
  return 1;
}

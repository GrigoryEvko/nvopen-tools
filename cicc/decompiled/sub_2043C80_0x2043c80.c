// Function: sub_2043C80
// Address: 0x2043c80
//
__int64 __fastcall sub_2043C80(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 result; // rax

  result = 0;
  if ( a3 == 1 )
  {
    result = 2;
    if ( *a2 != 105 )
      return 3 * (unsigned int)(*a2 == 109);
  }
  return result;
}

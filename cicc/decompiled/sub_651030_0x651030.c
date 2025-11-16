// Function: sub_651030
// Address: 0x651030
//
__int64 __fastcall sub_651030(_DWORD *a1)
{
  __int64 result; // rax
  __int16 v2; // dx

  *a1 = 0;
  if ( word_4F06418[0] != 56 )
    return 0;
  if ( !qword_4D04464 )
    return 0;
  v2 = sub_7BE840(0, 0);
  if ( !(_DWORD)qword_4D04464 || (result = 1, v2 != 152) )
  {
    if ( HIDWORD(qword_4D04464) && v2 == 83 )
    {
      *a1 = 1;
      return 1;
    }
    return 0;
  }
  return result;
}

// Function: sub_8CFC70
// Address: 0x8cfc70
//
__int64 __fastcall sub_8CFC70(__int64 *a1)
{
  __int64 result; // rax

  result = (__int64)&qword_4F074A0;
  if ( qword_4F074B0 == qword_4F60258 )
  {
    result = dword_4D03FC0;
    if ( dword_4D03FC0 )
    {
      if ( !a1[4] )
        return sub_8CBDE0(a1);
    }
  }
  return result;
}

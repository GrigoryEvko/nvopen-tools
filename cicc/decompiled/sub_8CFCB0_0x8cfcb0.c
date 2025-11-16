// Function: sub_8CFCB0
// Address: 0x8cfcb0
//
__int64 __fastcall sub_8CFCB0(__int64 a1)
{
  __int64 result; // rax

  result = (__int64)&qword_4F074A0;
  if ( qword_4F074B0 == qword_4F60258 )
  {
    result = dword_4D03FC0;
    if ( dword_4D03FC0 )
    {
      if ( !*(_QWORD *)(a1 + 32) )
        return sub_8CC930(a1, 0);
    }
  }
  return result;
}

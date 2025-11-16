// Function: sub_876960
// Address: 0x876960
//
__int64 __fastcall sub_876960(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax

  if ( qword_4D04900 )
    sub_8754F0(1, a1, a2);
  result = dword_4F04C3C;
  if ( !dword_4F04C3C )
  {
    if ( a4 )
      *(_QWORD *)(a4 + 72) = a3;
    else
      return sub_8699D0(a3, 29, 0);
  }
  return result;
}

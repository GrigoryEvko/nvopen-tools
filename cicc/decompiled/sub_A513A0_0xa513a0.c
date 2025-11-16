// Function: sub_A513A0
// Address: 0xa513a0
//
__int64 __fastcall sub_A513A0(unsigned int a1, __int64 a2)
{
  __int64 result; // rax

  if ( a1 == 3 )
    return sub_904010(a2, "thread_local(initialexec) ");
  if ( a1 > 3 )
  {
    if ( a1 == 4 )
      return sub_904010(a2, "thread_local(localexec) ");
  }
  else if ( a1 == 1 )
  {
    return sub_904010(a2, "thread_local ");
  }
  else if ( a1 == 2 )
  {
    return sub_904010(a2, "thread_local(localdynamic) ");
  }
  return result;
}

// Function: sub_1A517A0
// Address: 0x1a517a0
//
unsigned __int64 __fastcall sub_1A517A0(unsigned __int64 **a1)
{
  unsigned __int64 result; // rax
  unsigned __int64 v2; // r12

  result = (unsigned __int64)*a1;
  if ( ((unsigned __int8)*a1 & 4) != 0 )
  {
    result &= 0xFFFFFFFFFFFFFFF8LL;
    v2 = result;
    if ( result )
    {
      if ( *(_QWORD *)result != result + 16 )
        _libc_free(*(_QWORD *)result);
      return j_j___libc_free_0(v2, 48);
    }
  }
  return result;
}

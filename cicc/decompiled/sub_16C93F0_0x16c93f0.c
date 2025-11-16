// Function: sub_16C93F0
// Address: 0x16c93f0
//
__int64 __fastcall sub_16C93F0(_QWORD *a1)
{
  __int64 result; // rax

  if ( *a1 )
  {
    result = sub_16F05C0();
    if ( *a1 )
      return j_j___libc_free_0(*a1, 32);
  }
  return result;
}

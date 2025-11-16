// Function: sub_C33830
// Address: 0xc33830
//
__int64 __fastcall sub_C33830(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdi

  result = sub_C337D0(a1);
  if ( (unsigned int)result > 1 )
  {
    v2 = *(_QWORD *)(a1 + 8);
    if ( v2 )
      return j_j___libc_free_0_0(v2);
  }
  return result;
}

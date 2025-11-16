// Function: sub_16983A0
// Address: 0x16983a0
//
__int64 __fastcall sub_16983A0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdi

  result = sub_1698310(a1);
  if ( (unsigned int)result > 1 )
  {
    v2 = *(_QWORD *)(a1 + 8);
    if ( v2 )
      return j_j___libc_free_0_0(v2);
  }
  return result;
}

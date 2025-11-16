// Function: sub_103CCA0
// Address: 0x103cca0
//
__int64 __fastcall sub_103CCA0(__int64 a1)
{
  __int64 v1; // r12
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = 0;
  if ( v1 )
  {
    sub_103C970(v1);
    return j_j___libc_free_0(v1, 360);
  }
  return result;
}

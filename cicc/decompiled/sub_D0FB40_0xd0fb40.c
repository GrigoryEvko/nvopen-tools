// Function: sub_D0FB40
// Address: 0xd0fb40
//
__int64 __fastcall sub_D0FB40(__int64 a1)
{
  __int64 v1; // r12
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = 0;
  if ( v1 )
  {
    sub_D0FA70(v1);
    return j_j___libc_free_0(v1, 72);
  }
  return result;
}

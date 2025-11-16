// Function: sub_13976D0
// Address: 0x13976d0
//
__int64 __fastcall sub_13976D0(__int64 a1)
{
  __int64 v1; // r12
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = 0;
  if ( v1 )
  {
    sub_1397630(v1);
    return j_j___libc_free_0(v1, 72);
  }
  return result;
}

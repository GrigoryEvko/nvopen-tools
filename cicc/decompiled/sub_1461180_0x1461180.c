// Function: sub_1461180
// Address: 0x1461180
//
__int64 __fastcall sub_1461180(__int64 a1)
{
  __int64 v1; // r12
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = 0;
  if ( v1 )
  {
    sub_14602B0(v1);
    return j_j___libc_free_0(v1, 1040);
  }
  return result;
}

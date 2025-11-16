// Function: sub_DA2450
// Address: 0xda2450
//
__int64 __fastcall sub_DA2450(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 result; // rax

  v2 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = 0;
  if ( v2 )
  {
    sub_DA11D0(v2, a2);
    return j_j___libc_free_0(v2, 1576);
  }
  return result;
}

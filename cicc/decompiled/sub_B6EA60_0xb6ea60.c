// Function: sub_B6EA60
// Address: 0xb6ea60
//
__int64 __fastcall sub_B6EA60(__int64 *a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rdi

  result = *a1;
  v3 = *a2;
  *a2 = 0;
  v4 = *(_QWORD *)(result + 152);
  *(_QWORD *)(result + 152) = v3;
  if ( v4 )
    return j_j___libc_free_0(v4, 8);
  return result;
}

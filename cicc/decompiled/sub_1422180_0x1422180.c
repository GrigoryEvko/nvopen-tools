// Function: sub_1422180
// Address: 0x1422180
//
__int64 __fastcall sub_1422180(__int64 a1)
{
  __int64 v1; // r12
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = 0;
  if ( v1 )
  {
    sub_1421ED0(v1);
    return j_j___libc_free_0(v1, 344);
  }
  return result;
}

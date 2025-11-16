// Function: sub_36CDC10
// Address: 0x36cdc10
//
__int64 __fastcall sub_36CDC10(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi

  v2 = sub_22077B0(1u);
  v3 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v2;
  if ( v3 )
    j_j___libc_free_0(v3);
  return 0;
}

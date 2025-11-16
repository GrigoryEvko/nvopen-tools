// Function: sub_DF4DB0
// Address: 0xdf4db0
//
__int64 __fastcall sub_DF4DB0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rdi

  v2 = sub_22077B0(1);
  v3 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v2;
  if ( v3 )
    j_j___libc_free_0(v3, 1);
  return 0;
}

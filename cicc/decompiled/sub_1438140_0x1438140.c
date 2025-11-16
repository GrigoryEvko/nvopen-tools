// Function: sub_1438140
// Address: 0x1438140
//
__int64 __fastcall sub_1438140(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rdi

  v3 = sub_1632FA0(a2);
  v4 = sub_22077B0(16);
  if ( v4 )
    *(_QWORD *)(v4 + 8) = v3;
  v5 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v4;
  if ( v5 )
    j_j___libc_free_0(v5, 16);
  return 0;
}

// Function: sub_1499AA0
// Address: 0x1499aa0
//
__int64 __fastcall sub_1499AA0(__int64 a1)
{
  _QWORD *v2; // rax
  __int64 v3; // rdi

  v2 = (_QWORD *)sub_22077B0(8);
  if ( v2 )
    *v2 = 0;
  v3 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v2;
  if ( v3 )
    j_j___libc_free_0(v3, 8);
  return 0;
}

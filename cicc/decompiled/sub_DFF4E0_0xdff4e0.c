// Function: sub_DFF4E0
// Address: 0xdff4e0
//
__int64 __fastcall sub_DFF4E0(__int64 a1)
{
  _BYTE *v2; // rax
  __int64 v3; // rdi

  v2 = (_BYTE *)sub_22077B0(1);
  if ( v2 )
    *v2 = 0;
  v3 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v2;
  if ( v3 )
    j_j___libc_free_0(v3, 1);
  return 0;
}

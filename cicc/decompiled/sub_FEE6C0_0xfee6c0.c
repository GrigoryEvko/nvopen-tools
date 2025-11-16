// Function: sub_FEE6C0
// Address: 0xfee6c0
//
void __fastcall sub_FEE6C0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // rdi

  v2 = a1[2];
  while ( v2 )
  {
    v3 = v2;
    sub_FEE420(*(_QWORD **)(v2 + 24), a2);
    v4 = *(_QWORD *)(v2 + 40);
    v2 = *(_QWORD *)(v2 + 16);
    if ( v4 != v3 + 56 )
      _libc_free(v4, a2);
    a2 = 104;
    j_j___libc_free_0(v3, 104);
  }
}

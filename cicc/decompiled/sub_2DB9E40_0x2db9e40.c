// Function: sub_2DB9E40
// Address: 0x2db9e40
//
__int64 __fastcall sub_2DB9E40(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 *v3; // r12
  unsigned __int64 v4; // r13
  unsigned __int64 *v5; // rbx
  unsigned __int64 *v6; // r12
  unsigned __int64 v7; // rdi

  v2 = (__int64 *)sub_22077B0(0x110u);
  v3 = v2;
  if ( v2 )
    sub_2DB9DE0(v2, a2);
  v4 = *(_QWORD *)(a1 + 200);
  *(_QWORD *)(a1 + 200) = v3;
  if ( v4 )
  {
    v5 = *(unsigned __int64 **)(v4 + 64);
    v6 = &v5[6 * *(unsigned int *)(v4 + 72)];
    if ( v5 != v6 )
    {
      do
      {
        v6 -= 6;
        if ( (unsigned __int64 *)*v6 != v6 + 2 )
          _libc_free(*v6);
      }
      while ( v5 != v6 );
      v6 = *(unsigned __int64 **)(v4 + 64);
    }
    if ( v6 != (unsigned __int64 *)(v4 + 80) )
      _libc_free((unsigned __int64)v6);
    v7 = *(_QWORD *)(v4 + 8);
    if ( v7 != v4 + 24 )
      _libc_free(v7);
    j_j___libc_free_0(v4);
  }
  return 0;
}

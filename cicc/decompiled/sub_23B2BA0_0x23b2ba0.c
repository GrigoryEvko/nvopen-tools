// Function: sub_23B2BA0
// Address: 0x23b2ba0
//
void __fastcall sub_23B2BA0(__int64 a1)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  v2 = *(_QWORD *)(a1 + 8);
  v3 = v2 + 88LL * *(unsigned int *)(a1 + 16);
  if ( v2 != v3 )
  {
    do
    {
      v3 -= 88LL;
      v4 = *(_QWORD *)(v3 + 40);
      if ( v4 != v3 + 56 )
        j_j___libc_free_0(v4);
      v5 = *(_QWORD *)(v3 + 8);
      if ( v5 != v3 + 24 )
        j_j___libc_free_0(v5);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 8);
  }
  if ( v3 != a1 + 24 )
    _libc_free(v3);
}

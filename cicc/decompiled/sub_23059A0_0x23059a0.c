// Function: sub_23059A0
// Address: 0x23059a0
//
void __fastcall sub_23059A0(__int64 a1)
{
  __int64 v1; // r14
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  v1 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)a1 = &unk_4A15958;
  v3 = v1 + 8LL * *(unsigned int *)(a1 + 64);
  if ( v1 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(v3 - 8);
      v3 -= 8LL;
      if ( v4 )
      {
        v5 = *(_QWORD *)(v4 + 24);
        if ( v5 != v4 + 40 )
          _libc_free(v5);
        j_j___libc_free_0(v4);
      }
    }
    while ( v1 != v3 );
    v3 = *(_QWORD *)(a1 + 56);
  }
  if ( v3 != a1 + 72 )
    _libc_free(v3);
  v6 = *(_QWORD *)(a1 + 8);
  if ( v6 != a1 + 24 )
    _libc_free(v6);
}

// Function: sub_2B30DE0
// Address: 0x2b30de0
//
void __fastcall sub_2B30DE0(__int64 a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  v2 = *(_QWORD **)a1;
  v3 = (_QWORD *)(*(_QWORD *)a1 + 224LL * *(unsigned int *)(a1 + 8));
  if ( *(_QWORD **)a1 != v3 )
  {
    do
    {
      v3 -= 28;
      v4 = v3[23];
      if ( (_QWORD *)v4 != v3 + 25 )
        j_j___libc_free_0(v4);
      v5 = v3[19];
      if ( (_QWORD *)v5 != v3 + 21 )
        j_j___libc_free_0(v5);
      v6 = v3[1];
      if ( (_QWORD *)v6 != v3 + 3 )
        _libc_free(v6);
    }
    while ( v3 != v2 );
    v3 = *(_QWORD **)a1;
  }
  if ( v3 != (_QWORD *)(a1 + 16) )
    _libc_free((unsigned __int64)v3);
}

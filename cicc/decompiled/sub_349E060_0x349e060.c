// Function: sub_349E060
// Address: 0x349e060
//
void __fastcall sub_349E060(_QWORD *a1)
{
  _QWORD *v1; // rbx
  unsigned __int64 v2; // r12
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = (unsigned __int64)v1;
      sub_349E060(v1[3]);
      v3 = v1[52];
      v1 = (_QWORD *)v1[2];
      if ( v3 != v2 + 432 )
        _libc_free(v3);
      v4 = *(_QWORD *)(v2 + 368);
      if ( v4 != v2 + 384 )
        _libc_free(v4);
      v5 = *(_QWORD *)(v2 + 96);
      if ( v5 != v2 + 112 )
        _libc_free(v5);
      j_j___libc_free_0(v2);
    }
    while ( v1 );
  }
}

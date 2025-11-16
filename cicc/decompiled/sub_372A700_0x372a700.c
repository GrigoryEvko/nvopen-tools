// Function: sub_372A700
// Address: 0x372a700
//
void __fastcall sub_372A700(_QWORD *a1)
{
  _QWORD *v1; // r12
  unsigned __int64 v2; // r13
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = (unsigned __int64)v1;
      sub_372A700(v1[3]);
      v3 = v1[11];
      v1 = (_QWORD *)v1[2];
      while ( v3 )
      {
        sub_372A530(*(_QWORD *)(v3 + 24));
        v4 = v3;
        v3 = *(_QWORD *)(v3 + 16);
        j_j___libc_free_0(v4);
      }
      v5 = *(_QWORD *)(v2 + 48);
      if ( v5 != v2 + 64 )
        _libc_free(v5);
      j_j___libc_free_0(v2);
    }
    while ( v1 );
  }
}

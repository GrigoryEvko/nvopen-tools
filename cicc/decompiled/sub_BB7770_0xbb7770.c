// Function: sub_BB7770
// Address: 0xbb7770
//
__int64 __fastcall sub_BB7770(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rdi
  _QWORD *v5; // rdi

  v3 = a1[8];
  *a1 = &unk_49DACB8;
  while ( v3 )
  {
    sub_BB7530(*(_QWORD *)(v3 + 24));
    v4 = v3;
    v3 = *(_QWORD *)(v3 + 16);
    a2 = 40;
    j_j___libc_free_0(v4, 40);
  }
  v5 = (_QWORD *)a1[2];
  if ( v5 != a1 + 4 )
    _libc_free(v5, a2);
  return j_j___libc_free_0(a1, 96);
}

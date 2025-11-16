// Function: sub_1E785F0
// Address: 0x1e785f0
//
void *__fastcall sub_1E785F0(_QWORD *a1)
{
  _QWORD *v1; // r13
  _QWORD *v3; // rbx
  _QWORD *v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rbx
  __int64 v7; // rdi
  unsigned __int64 v8; // rdi

  v1 = a1 + 70;
  v3 = (_QWORD *)a1[70];
  *a1 = off_49FCBB0;
  if ( a1 + 70 != v3 )
  {
    do
    {
      v4 = v3;
      v3 = (_QWORD *)*v3;
      j_j___libc_free_0(v4, 40);
    }
    while ( v1 != v3 );
  }
  v5 = a1[66];
  if ( v5 )
    j_j___libc_free_0(v5, a1[68] - v5);
  j___libc_free_0(a1[63]);
  v6 = a1[58];
  while ( v6 )
  {
    sub_1E783A0(*(_QWORD *)(v6 + 24));
    v7 = v6;
    v6 = *(_QWORD *)(v6 + 16);
    j_j___libc_free_0(v7, 48);
  }
  v8 = a1[38];
  if ( (_QWORD *)v8 != a1 + 40 )
    _libc_free(v8);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}

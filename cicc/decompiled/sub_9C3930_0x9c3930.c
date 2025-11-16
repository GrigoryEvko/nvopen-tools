// Function: sub_9C3930
// Address: 0x9c3930
//
__int64 __fastcall sub_9C3930(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // rbx
  _QWORD *v4; // r13
  _QWORD *v5; // rdi

  v3 = a1 + 7;
  v4 = (_QWORD *)a1[7];
  *a1 = &unk_49D97D0;
  if ( v4 )
  {
    if ( *v4 )
      j_j___libc_free_0(*v4, v4[2] - *v4);
    a2 = 24;
    j_j___libc_free_0(v4, 24);
  }
  v5 = (_QWORD *)a1[5];
  if ( v3 != v5 )
    _libc_free(v5, a2);
  return j_j___libc_free_0(a1, 72);
}

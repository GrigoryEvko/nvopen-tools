// Function: sub_1DDBE30
// Address: 0x1ddbe30
//
void *__fastcall sub_1DDBE30(_QWORD *a1)
{
  __int64 v2; // rdi
  _QWORD *v3; // rbx
  void *result; // rax
  _QWORD *v5; // r12
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __int64 v9; // rdi
  _QWORD *i; // r12
  _QWORD *v11; // rdi
  __int64 v12; // rdi

  *a1 = &unk_49FB1C8;
  j___libc_free_0(a1[21]);
  v2 = a1[17];
  if ( v2 )
    j_j___libc_free_0(v2, a1[19] - v2);
  v3 = (_QWORD *)a1[11];
  result = &unk_49E8A50;
  for ( *a1 = &unk_49E8A50; a1 + 11 != v3; result = (void *)j_j___libc_free_0(v5, 192) )
  {
    v5 = v3;
    v3 = (_QWORD *)*v3;
    v6 = v5[18];
    if ( (_QWORD *)v6 != v5 + 20 )
      _libc_free(v6);
    v7 = v5[14];
    if ( (_QWORD *)v7 != v5 + 16 )
      _libc_free(v7);
    v8 = v5[4];
    if ( (_QWORD *)v8 != v5 + 6 )
      _libc_free(v8);
  }
  v9 = a1[8];
  if ( v9 )
    result = (void *)j_j___libc_free_0(v9, a1[10] - v9);
  for ( i = (_QWORD *)a1[5]; a1 + 5 != i; result = (void *)j_j___libc_free_0(v11, 40) )
  {
    v11 = i;
    i = (_QWORD *)*i;
  }
  v12 = a1[1];
  if ( v12 )
    return (void *)j_j___libc_free_0(v12, a1[3] - v12);
  return result;
}

// Function: sub_1DDBF60
// Address: 0x1ddbf60
//
__int64 __fastcall sub_1DDBF60(_QWORD *a1)
{
  __int64 v2; // rdi
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // rdi
  _QWORD *v9; // r12
  _QWORD *v10; // rdi
  __int64 v11; // rdi

  *a1 = &unk_49FB1C8;
  j___libc_free_0(a1[21]);
  v2 = a1[17];
  if ( v2 )
    j_j___libc_free_0(v2, a1[19] - v2);
  v3 = (_QWORD *)a1[11];
  *a1 = &unk_49E8A50;
  while ( a1 + 11 != v3 )
  {
    v4 = v3;
    v3 = (_QWORD *)*v3;
    v5 = v4[18];
    if ( (_QWORD *)v5 != v4 + 20 )
      _libc_free(v5);
    v6 = v4[14];
    if ( (_QWORD *)v6 != v4 + 16 )
      _libc_free(v6);
    v7 = v4[4];
    if ( (_QWORD *)v7 != v4 + 6 )
      _libc_free(v7);
    j_j___libc_free_0(v4, 192);
  }
  v8 = a1[8];
  if ( v8 )
    j_j___libc_free_0(v8, a1[10] - v8);
  v9 = (_QWORD *)a1[5];
  while ( a1 + 5 != v9 )
  {
    v10 = v9;
    v9 = (_QWORD *)*v9;
    j_j___libc_free_0(v10, 40);
  }
  v11 = a1[1];
  if ( v11 )
    j_j___libc_free_0(v11, a1[3] - v11);
  return j_j___libc_free_0(a1, 192);
}

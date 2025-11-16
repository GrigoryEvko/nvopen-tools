// Function: sub_12F0C20
// Address: 0x12f0c20
//
void __fastcall sub_12F0C20(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // rdi
  _QWORD *v4; // rdi
  _QWORD *v5; // rdi
  __int64 v6; // rdi

  *a1 = &unk_49EEBF0;
  v3 = (_QWORD *)a1[31];
  if ( v3 != a1 + 33 )
  {
    a2 = a1[33] + 1LL;
    j_j___libc_free_0(v3, a2);
  }
  v4 = (_QWORD *)a1[25];
  a1[24] = &unk_49E7488;
  if ( v4 != a1 + 27 )
  {
    a2 = a1[27] + 1LL;
    j_j___libc_free_0(v4, a2);
  }
  v5 = (_QWORD *)a1[20];
  if ( v5 != a1 + 22 )
  {
    a2 = a1[22] + 1LL;
    j_j___libc_free_0(v5, a2);
  }
  v6 = a1[12];
  if ( v6 != a1[11] )
    _libc_free(v6, a2);
}
